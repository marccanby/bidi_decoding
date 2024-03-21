import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig, CrossEntropyCriterion
try:
    from examples.bidi_decoding.bidi_utils import  reverseSequence, createGatherers, compute_advanced_loss
except:
    from bidi_decoding.bidi_utils import reverseSequence, createGatherers, \
        compute_advanced_loss

import torch.nn.functional as F


@dataclass
class BidiCriterionConfig(CrossEntropyCriterionConfig):
    loss_style: str = field(
        default='sep',
        metadata={
            "help": "Loss style (sep, sep-rand, MML)"
        },
    )

    order_temp: float = field (
        default=1.0,
        metadata={
            'help': 'temp to smooth the order probabilities in MML case'
        }
    )



@register_criterion(
    "bidi_loss", dataclass=BidiCriterionConfig
)
class BidiCriterion(CrossEntropyCriterion):
    def __init__(self, task, sentence_avg, loss_style, order_temp):
        super().__init__(task, sentence_avg)
        self.loss_style = loss_style
        self.order_temp = order_temp
        assert self.loss_style in {'sep', 'MML', 'sep-rand'}

        if self.loss_style in {'sep', 'sep-rand'}:
            self.order_temp = None

    def forward(self, model, sample, reduce=True, optimizer=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if model.task.rerank_dictionary is not None:
            return 0, sample["target"].size(0) if self.sentence_avg else sample["ntokens"], {
            "loss": 0,
            "loss_l2r": 0,
            "loss_r2l": 0,
            "loss_join": 0,
            "loss_mml": 0,
            "loss_order": 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample["target"].size(0) if self.sentence_avg else sample["ntokens"],
        }

        assert model.tp == 'fullbidi'
        assert self.task.need_extra_spec

        warmup = self.task.trainer.lr_scheduler.cfg['warmup_updates']
        assert warmup % 2 == 0 # Need to check whether this is still needed - don't think so.

        src_tokens, src_lengths, inputs_l2r, trg_l2r = sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'], sample['target']
        inputs_r2l, trg_r2l = reverseSequence(inputs_l2r, trg_l2r)
        if self.loss_style in  {'sep', 'sep-rand'}: assert not model.do_order
        else: assert model.do_order
        lengths = torch.sum(inputs_l2r != self.task.target_dictionary.pad_index, dim=1)+1

        assert torch.all(torch.gather(trg_l2r, 1,
                                      index=((trg_l2r != self.task.target_dictionary.pad()).sum(dim=1)-1).unsqueeze(1)) == self.task.target_dictionary.eos_index) and \
               torch.all(torch.gather(trg_r2l, 1, index=((trg_r2l != self.task.target_dictionary.pad()).sum(dim=1)-1).unsqueeze(1)) == self.task.target_dictionary.eos_index)

        gath = [createGatherers(length.item()) for length in lengths]
        gath0 = [q[0] for q in gath]
        gath_concat = torch.cat(gath0, dim=0)

        tok_l2r, tok_r2l, join_probs, order_probs = model(src_tokens, src_lengths, inputs_l2r, inputs_r2l, self.task.target_dictionary, do_rand = self.loss_style == 'sep-rand' and (model.training))
        if self.loss_style != 'sep-rand' or self.loss_style == 'sep-rand' and not model.training: assert len(join_probs.shape) == 2, "for training, shoyuld only output 2 things"
        else: assert len(join_probs) == 2 and len(join_probs[0].shape) == 2 # it's a tuple here
        if self.loss_style in {'sep', 'sep-rand'}: assert order_probs is None
        if self.loss_style == 'sep-rand' and model.training:
            _, idxes_h = join_probs

        if not model.training or self.loss_style in {'sep', 'sep-rand'}:
            # Always doing no eos now # don't think need to touch tokl2r or r2l since we ignore pad
            trg_l2r.scatter_(1,
                             index=((trg_l2r != self.task.target_dictionary.pad()).sum(dim=1) - 1).unsqueeze(1),
                             src=torch.full((trg_l2r.shape[0], 1), self.task.target_dictionary.pad_index,
                                            device=trg_l2r.device))
            trg_r2l.scatter_(1,
                             index=((trg_r2l != self.task.target_dictionary.pad()).sum(dim=1) - 1).unsqueeze(1),
                             src=torch.full((trg_r2l.shape[0], 1), self.task.target_dictionary.pad_index,
                                            device=trg_r2l.device))

            left_idxes, right_idxes = gath_concat[:, 0], gath_concat[:, 1]
            gath_wd_idxes = torch.cat(
                [torch.full((len(gath0[i]),), i, device=inputs_l2r.device) for i in range(len(gath0))])
            orig_trg_l2r, orig_trg_r2l = trg_l2r, trg_r2l # trick, since in validation this is needed in its original form for down below (in training MML, it doesnt enter here; in training sep/srand, it doesnt enter below)
            trg_l2r = trg_l2r[gath_wd_idxes.long(), left_idxes.long()].unsqueeze(1)
            trg_r2l = trg_r2l[gath_wd_idxes.long(), right_idxes.long()].unsqueeze(1)
            tok_l2r, tok_r2l = tok_l2r.unsqueeze(0), tok_r2l.unsqueeze(
                0)  # not sure why have to do this; i think noramlly, the 0th dim is the bs; but here that has been removed, so trickily adding it back so compute_loss is happy
            if self.loss_style == 'sep-rand' and model.training:
                trg_l2r = trg_l2r[idxes_h.long(),:]
                trg_r2l = trg_r2l[idxes_h.long(), :]


            loss_l2r, _ = self.compute_loss(model, tok_l2r, trg_l2r, True, reduce=reduce)
            loss_r2l, _ = self.compute_loss(model, tok_r2l, trg_r2l, True, reduce=reduce)

            tok_l2r, tok_r2l = tok_l2r.squeeze(0), tok_r2l.squeeze(0) # squeeze it back to how it was. in validation loss, we both enter here and the MML section below, so it had to be right shape for here and then fixed again
            trg_l2r, trg_r2l = orig_trg_l2r, orig_trg_r2l # same idea. note that here the scattering thing was done (and not during training), but dont think should be that big of dedal (may not even matter at all)
        else: loss_l2r, loss_r2l = 0, 0

        gold_join = torch.cat([x[1] for x in gath], dim=0).to(tok_l2r.device)

        # orders: 0 is left, 1 is right
        # joins: 0 is not join, 1 is join
        if not model.training or self.loss_style == 'sep':
            loss_join, _ = self.compute_loss(model, join_probs, gold_join, False, reduce=reduce, weight=None)
        else: loss_join = 0

        if not model.training and model.do_order:
            normmed_probs = model.get_normalized_probs(order_probs, log_probs=True)
            true_order_probs = torch.full_like(normmed_probs, 0.5, device = normmed_probs.device)
            loss_order = torch.sum(torch.sum(-true_order_probs * normmed_probs, 1))
            assert reduce # if true, have to sum (which i'm doing)
        else: loss_order = 0

        if self.loss_style == 'sep-rand' and model.training:
            join_probs, join_idxes = join_probs
            gold_join = gold_join[join_idxes.long()]
            loss_join, _ = self.compute_loss(model, join_probs, gold_join, False, reduce=reduce) # i think this method would also work on the sep case and might be cleaner to have it done the same way

        if self.loss_style == 'MML':
            assert tok_l2r.shape[0] == join_probs.shape[0] and tok_r2l.shape[0] == join_probs.shape[0]
            gath0 = [g[0] for g in gath]
            gath_concat = torch.cat(gath0, dim=0)
            gath_wd_idxes = torch.cat(
                [torch.full((len(gath0[i]),), i, device=tok_l2r.device) for i in range(len(gath0))])
            if self.loss_style in {'sep', 'sep-rand'}:# or not model.do_order: # last case from my testing of adding in do_order later (so be careful!)
                order_probs = torch.zeros_like(join_probs).fill_(math.log(0.5))
                no_order_softmax=True
            else: no_order_softmax=False

            tl2r, tr2l = tok_l2r, tok_r2l

            if not model.training or self.loss_style == 'MML':
                loss_mml = self.compute_advanced_loss(model, tl2r, trg_l2r, tr2l, trg_r2l, join_probs, gold_join,
                                           order_probs, gath_concat, gath_wd_idxes, reduce, no_order_softmax)
            else: loss_mml = 0
        else: loss_mml = 0

        if 'sep' in self.loss_style:
            loss = (loss_l2r + loss_r2l + loss_join) / 3 + (0 * order_probs.sum() if order_probs is not None else 0)
        elif self.loss_style == 'MML':
            loss = loss_mml
        else: assert False

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "loss_l2r": loss_l2r.data if type(loss_l2r) != int else loss_l2r,
            "loss_r2l": loss_r2l.data if type(loss_r2l) != int else loss_r2l,
            "loss_join": loss_join.data if type(loss_join) != int else loss_join,
            "loss_mml": loss_mml.data if type(loss_mml) != int else loss_mml,
            "loss_order": loss_order.data if type(loss_order) != int else loss_order,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, target, ignore_pad, reduce=True, weight=None):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # takes log softmax xs last dim, so fine
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx if ignore_pad else -9999999999999,
            reduction="sum" if reduce else "none",
            weight=weight
        )
        return loss, loss

    def compute_advanced_loss(self, model, tok_l2r, trg_l2r, tok_r2l, trg_r2l, join_probs, gold_join, order_probs,
                              gath_concat, gath_wd_idxes, reduce, no_order_softmax):

        probs_l2r = model.get_normalized_probs(tok_l2r, log_probs=True) # takes log softmax xs last dim, so fine
        probs_r2l = model.get_normalized_probs(tok_r2l, log_probs=True)  # takes log softmax xs last dim, so fine
        probs_join = model.get_normalized_probs(join_probs, log_probs=True)  # takes log softmax xs last dim, so fine
        if not no_order_softmax:
            if self.order_temp != 1:
                nsteps = self.task.trainer.get_num_updates()
                warmup = self.task.trainer.lr_scheduler.cfg['warmup_updates']
                a=2 # originally a = 0.,5, super steep
                if nsteps >= warmup: otemp = 1
                else: otemp = (self.order_temp - 1) / (warmup ** a) * ((-(nsteps - warmup)) ** a) + 1
                assert otemp >= 1 and otemp <= self.order_temp
            else: otemp = 1
            probs_order = F.log_softmax(order_probs/otemp, dim=-1) # takes log softmax xs last dim, so fine
        else:
            probs_order = order_probs # in case of 0.5, 0.5

        return compute_advanced_loss(probs_l2r, trg_l2r, probs_r2l, trg_r2l, probs_join, gold_join, probs_order,
                          gath_concat, gath_wd_idxes, reduce, False)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l2r_sum = sum(log.get("loss_l2r", 0) for log in logging_outputs)
        loss_r2l_sum = sum(log.get("loss_r2l", 0) for log in logging_outputs)
        loss_join_sum = sum(log.get("loss_join", 0) for log in logging_outputs)
        loss_order_sum = sum(log.get("loss_order", 0) for log in logging_outputs)
        loss_mml_sum = sum(log.get("loss_mml", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_l2r", loss_l2r_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_r2l", loss_r2l_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_join", loss_join_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_order", loss_order_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mml", loss_mml_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
