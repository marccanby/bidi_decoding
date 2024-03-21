from fairseq.sequence_generator import SequenceGenerator
import math
from typing import Dict, List, Optional, Tuple
import sys, random

import torch
import torch.nn as nn
from fairseq import utils
from torch import Tensor

try:
    from examples.bidi_decoding.bidi_utils import reverseSequence, createGatherers, compute_advanced_loss, subsetTensorToSpecificIdxes
except:
    from bidi_decoding.bidi_utils import reverseSequence, createGatherers, compute_advanced_loss, subsetTensorToSpecificIdxes


class BidiEnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        assert len(models) == 1
        # method '__len__' is not supported in ModuleList for torch script
        self.model = models[0]
        assert self.model.tp == 'fullbidi'
        self.has_incremental: bool = False

    def forward(self):
        pass

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models if hasattr(m, "max_decoder_positions")] + [sys.maxsize])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        return self.model.model.encoder.forward_torchscript(net_input)

    @torch.jit.export
    def forward_decoder(
        self,
        tokens_l2r,
        tokens_r2l,
        pad_idx,
        encoder_outs: Tuple[List[Dict[str, List[Tensor]]]],
        temperature: float = 1.0
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None

        # decode each model
        order_probs=None

        predictor_idxes_l2r = (tokens_l2r != pad_idx).sum(dim=1) - 1
        predictor_idxes_r2l = (tokens_r2l != pad_idx).sum(dim=1) - 1

        cols_to_keep_l2r = ((tokens_l2r != pad_idx).sum(dim=0) > 0)
        cols_to_keep_r2l = ((tokens_r2l != pad_idx).sum(dim=0) > 0)

        tokens_l2r, tokens_r2l = tokens_l2r[:, cols_to_keep_l2r], tokens_r2l[:, cols_to_keep_r2l]

        sep1 = self.model.trg_dictionary.indices['<sep>']
        sep2 = self.model.trg_dictionary.indices['<sep2>']
        clsj = self.model.trg_dictionary.indices['<cls_j>']
        clso = self.model.trg_dictionary.indices['<cls_o>']

        if not torch.all((predictor_idxes_l2r == 0 ) & (predictor_idxes_r2l == 0)):
            l2r_side=subsetTensorToSpecificIdxes(tokens_l2r, predictor_idxes_l2r, -1)
            r2l_side = subsetTensorToSpecificIdxes(tokens_r2l, predictor_idxes_r2l, -1)
            if not self.model.do_order:
                l2r_side =  torch.cat([torch.tensor(clsj, device=tokens_l2r.device).unsqueeze(0).repeat(l2r_side.shape[0],1),
                                           l2r_side],
                                          dim=1)
            else:
                l2r_side = torch.cat([torch.tensor(clsj, device=tokens_l2r.device).unsqueeze(0).repeat(l2r_side.shape[0],1),
                                      torch.tensor(clso, device=tokens_l2r.device).unsqueeze(0).repeat(l2r_side.shape[0], 1),
                                      l2r_side], dim=1)
            size = (predictor_idxes_l2r + predictor_idxes_r2l)[0].item() + 2 + 2 + 1 + (1 if self.model.do_order else 0)
            new_chunk = torch.full((l2r_side.shape[0], size), -1, device=l2r_side.device)
            new_chunk[:,:l2r_side.shape[1]] = l2r_side
            first_neg_one = (new_chunk != -1).sum(dim=1)
            new_chunk.scatter_(1, first_neg_one.unsqueeze(1), sep1)
            new_chunk.scatter_(1, first_neg_one.unsqueeze(1)+1, sep2)
            r2l_flip = torch.flip(r2l_side, dims=[1]) # FLIP
            new_chunk[new_chunk == -1] = r2l_flip[r2l_flip != -1].long()
            assert torch.all(new_chunk != -1)

            idxes_l2r = (l2r_side != -1).sum(dim=1)

        else:
            if not self.model.do_order:
                new_chunk = torch.cat([torch.tensor(clsj, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       tokens_l2r,
                                       torch.tensor(sep1, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       torch.tensor(sep2, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       tokens_r2l],
                                      dim=1)
                idx_l2r =  1 + (1 if self.model.do_order else 0) + tokens_l2r.shape[1]
                assert idx_l2r == 2
                idxes_l2r = torch.full((new_chunk.shape[0],), fill_value=idx_l2r, device=new_chunk.device)
            else:
                new_chunk = torch.cat([torch.tensor(clsj, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       torch.tensor(clso, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0], 1),
                                       tokens_l2r,
                                       torch.tensor(sep1, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       torch.tensor(sep2, device=tokens_l2r.device).unsqueeze(0).repeat(tokens_l2r.shape[0],1),
                                       tokens_r2l],
                                      dim=1)
                idx_l2r = 1 + (1 if self.model.do_order else 0) + tokens_l2r.shape[1]
                assert idx_l2r == 3
                idxes_l2r = torch.full((new_chunk.shape[0],), fill_value=idx_l2r, device=new_chunk.device)

        result_here = self.model.model.decoder(new_chunk,
                                         encoder_out=encoder_outs,
                                         features_only=True,
                                         incremental_state=None,
                                               full_context_alignment=True, # SUPER IMPORTNAT - dont want causal mask
                                               )
        lin_result_for_l2r = self.model.model.decoder.output_layer(result_here[0])
        lin_result_for_r2l = self.model.r2l_lin_layer(result_here[0])
        lin_result_for_join = self.model.join_lin_layer(result_here[0])  # ~bsz x L x 2
        if self.model.do_order: lin_result_for_order = self.model.order_lin_layer(result_here[0])

        # Now have to get the right idxes for the various things
        join_probs = lin_result_for_join[:, 0, :]
        if self.model.do_order: order_probs = lin_result_for_order[:, 1, :]
        idxes_r2l = 1+idxes_l2r
        l2r_out = torch.gather(lin_result_for_l2r, 1,
                                          idxes_l2r.unsqueeze(1).repeat(1, lin_result_for_l2r.shape[2]).unsqueeze(
                                              1).long()).squeeze(1)
        r2l_out = torch.gather(lin_result_for_r2l, 1,
                                          idxes_r2l.unsqueeze(1).repeat(1, lin_result_for_r2l.shape[2]).unsqueeze(
                                              1).long()).squeeze(1)

        attn: Optional[Tensor] = None # can find original code in origainl generator, if wanna keep attn

        assert temperature == 1.0, 'havent thought if it messes anything up if temp != 1'
        l2r_out, r2l_out = l2r_out.div_(temperature), r2l_out.div_(temperature)

        probs_l2r = self.model.get_normalized_probs(l2r_out, log_probs=True, sample=None)
        probs_r2l = self.model.get_normalized_probs(r2l_out, log_probs=True, sample=None)
        joins = join_probs
        joins = self.model.get_normalized_probs(joins, log_probs=True, sample=None).squeeze(1)
        if order_probs is not None:
            orders = self.model.get_normalized_probs(order_probs, log_probs=True, sample=None).squeeze(1)
        else:
            orders = torch.zeros_like(joins).fill_(math.log(0.5))

        return (probs_l2r, probs_r2l, joins, orders), attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        assert encoder_outs is not None
        return self.model.model.encoder.reorder_encoder_out(encoder_outs, new_order)




class BidiSequenceGenerator(SequenceGenerator):

    def __init__(self, models, tgt_dict, **kwargs):
        super().__init__(models, tgt_dict, **kwargs)
        self.model = BidiEnsembleModel(models)

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        assert not self.model.has_incremental_states()
        net_input = sample["net_input"]

        if self.model.model.task.rerank_test:
            assert self.model.model.task.rerank_dictionary is not None
        if self.model.model.task.rerank_dictionary is not None and self.model.model.task.is_evaluating_test:
            finalized = []
            for i in range(len(sample['id'])):
                res = self.model.model.task.rerank_dictionary[sample['id'][i]]
                assert res[0] == net_input['src_tokens'][i,:][net_input['src_tokens'][i,:] != self.pad].tolist()
                finalized.append([
                    {'tokens': torch.Tensor(x).to(net_input['src_tokens'].device),
                     'score': torch.Tensor([-1]).squeeze().to(net_input['src_tokens'].device)} for x in res[1]
                ])
                assert len(finalized[-1]) == len(res[1])
                try:
                    assert len(res[1]) <= self.beam_size
                except:
                    print(res)
                    print(''.join([self.tgt_dict[q] for q in res[0]]))
                    for xxxxxx in res[1]:
                        print(''.join([self.tgt_dict[q] for q in xxxxxx]))
                    print(finalized[-1])
                    assert len(res[1]) <= self.beam_size # so it still fails
            self.runMmlReranker(net_input, finalized, bos_token)

            # Sort by score descending
            for sent in range(len(finalized)):
                scores = torch.tensor(
                    [float(elem["score"].item()) for elem in finalized[sent]]
                )
                _, sorted_scores_indices = torch.sort(scores, descending=True)
                finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
                finalized[sent] = torch.jit.annotate(
                    List[Dict[str, Tensor]], finalized[sent]
                )
            return finalized



        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        else: assert False, 'removed these other cases'

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        def make_tokens():
            tokens = (
                torch.zeros(bsz * beam_size, 1)
                    .to(src_tokens)
                    .long()
                    .fill_(self.pad)
            )  # +2 for eos and pad
            tokens[:, 0] = self.eos if bos_token is None else bos_token
            return tokens
        tokens_l2r, tokens_r2l = make_tokens(), make_tokens()
        final_directions_l2r, final_directions_r2l = torch.zeros(bsz * beam_size, 1, device=tokens_l2r.device).long(), torch.zeros(bsz * beam_size, 1, device=tokens_l2r.device).long()
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens_l2r)
                .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens_l2r).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens_l2r)

        tokens_l2r_last, tokens_r2l_last = None, None # lazy way to figure out which direction it went
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            with torch.autograd.profiler.record_function("EnsembleModel: forward_decoder"):
                (probs_l2r, probs_r2l, joins, orders), avg_attn_scores = self.model.forward_decoder(
                    tokens_l2r,
                    tokens_r2l,
                    self.pad,
                    encoder_outs,
                    self.temperature
                )

            assert self.lm_model is None

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                assert False
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                joins[:,1] = -math.inf #lprobs[:, self.eos] = -math.inf # join # 1 is join

            # not sure why this line is here, seems useless but harmless
            probs_l2r[probs_l2r != probs_l2r] = torch.tensor(-math.inf).to(probs_l2r)
            probs_r2l[probs_r2l != probs_r2l] = torch.tensor(-math.inf).to(probs_r2l)

            if self.model.model.task.force_l2r:
                probs_r2l[:,:] = -math.inf
            if self.model.model.task.force_r2l:
                probs_l2r[:,:] = -math.inf

            probs_l2r[:, self.pad] = -math.inf  # never select pad
            probs_r2l[:, self.pad] = -math.inf
            probs_l2r[:, self.unk] -= self.unk_penalty  # apply unk penalty
            probs_r2l[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # marc: i added these, since in my situation we never predict eos or bos (seems it assumes 0 for this), rather we predict join
            # esp. unsure about the 0 (bos) case
            probs_l2r[:, self.eos] = -math.inf
            probs_r2l[:, self.eos] = -math.inf
            probs_l2r[:, 0] = -math.inf
            probs_r2l[:, 0] = -math.inf

            # adding some special ones in now:
            more_to_prevent = set([])
            if '<sep>' in self.tgt_dict.indices: more_to_prevent.add('<sep>')
            try: # In case tagset does not exist
                for xyz in self.tgt_dict.tagset: more_to_prevent.add(xyz)
            except:
                pass

            for xyz in more_to_prevent:
                idxx = self.tgt_dict.indices[xyz]
                probs_l2r[:,idxx] = -math.inf
                probs_r2l[:,idxx] = -math.inf


            # handle max length constraint
            if step >= max_len:
                probs_l2r[:,:] = -math.inf
                probs_r2l[:,:] = -math.inf
                joins[:,0] = -math.inf # 0 is not join

            # Now combine the tensors - NOTE: if step==0, join=0 since min len = 1
            assert self.min_len >= 1
            probs_l2r = joins[:, 0].unsqueeze(1) + orders[:, 0].unsqueeze(1) + probs_l2r # 0 is left, 0 is not join
            probs_r2l = joins[:, 0].unsqueeze(1) + orders[:, 1].unsqueeze(1) + probs_r2l # 1 is right, 0 is not join
            overall_probs = torch.cat([joins[:,1].unsqueeze(1), probs_l2r, probs_r2l], dim=1) # 1 is join

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None: assert False, 'havent done yet'

            scores = scores.type_as(probs_l2r)
            eos_bbsz_idx = torch.empty(0).to(
                tokens_l2r
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths: assert False, 'havnent done yet'

            if self.repeat_ngram_blocker is not None: assert False, 'ahvent done yet'

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                overall_probs.view(bsz, -1, overall_probs.shape[1]),#lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                None, #  appears these arent used, so giving them None
                None,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            # me: 0 for join
            eos_mask = cand_indices.eq(0) & cand_scores.ne(-math.inf) #cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens_l2r,
                    tokens_r2l,
                    final_directions_l2r,
                    final_directions_r2l,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_l2r = tokens_l2r.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_r2l = tokens_r2l.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

                final_directions_l2r = final_directions_l2r.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                final_directions_r2l = final_directions_r2l.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)

                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # I am going to convert cand_indices here to the actual situation; but do have to be careful taht doesn't mess anything up in the rest of the fxn.
            assert cand_indices.shape == cand_scores.shape and cand_indices.shape == cand_beams.shape
            shape = tuple(cand_indices.shape)

            msk_join, msk_l2r, msk_r2l = cand_indices == 0, (1 <= cand_indices) & (cand_indices <= self.vocab_size), self.vocab_size + 1 <= cand_indices
            assert self.eos == 2 and self.pad == 1 and self.unk == 3, 'may have to check next lines if not'
            assert torch.all(cand_indices[msk_l2r] - 1 >= 3) # these conditions are sanity check that it predicts an actual word; may be different if more special symbols
            assert torch.all(cand_indices[msk_r2l] - 1 - self.vocab_size >= 3)

            # note: doing it this way puts the correct idx in each spot, but can no longer tell from it which is which. hence need directions
            cand_indices[msk_join], cand_indices[msk_l2r], cand_indices[msk_r2l] = cand_indices[msk_join], cand_indices[msk_l2r] - 1, cand_indices[msk_r2l] - 1 - self.vocab_size
            assert torch.all(cand_indices < self.vocab_size)
            # note: i believe a 0 in cand_indices can either refer to JOIN or to the first thing in vocab, but don't think problem, since never use 0 in either case (join doesn't touch this, and 0 in vocab is illegal)

            directions = torch.full(shape, -1, device=cand_indices.device)
            directions[msk_join], directions[msk_l2r], directions[msk_r2l] = 0, 1, 2
            assert torch.all(directions >= 0)

            # also, want cand_indices_l2r and _r2l that tells me the correct index for the ones going in that direction, else padding
            # these should basically be padding in opposite places
            cand_indices_l2r, cand_indices_r2l = torch.full(shape, self.pad, device=cand_indices.device), torch.full(shape, self.pad, device=cand_indices.device)
            cand_indices_l2r[msk_l2r], cand_indices_r2l[msk_r2l] = cand_indices[msk_l2r], cand_indices[msk_r2l]

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens_l2r = torch.cat([tokens_l2r, torch.full((tokens_l2r.shape[0], 1), self.pad, device=tokens_l2r.device)], dim=1)
            tokens_r2l = torch.cat([tokens_r2l, torch.full((tokens_r2l.shape[0], 1), self.pad, device=tokens_r2l.device)], dim=1)
            final_directions_l2r = torch.cat(
                [final_directions_l2r, torch.zeros((final_directions_l2r.shape[0], 1), device=tokens_l2r.device).long()], dim=1)
            final_directions_r2l = torch.cat(
                [final_directions_r2l, torch.zeros((final_directions_r2l.shape[0], 1), device=tokens_r2l.device).long()], dim=1)

            tokens_l2r[:, : step + 1] = torch.index_select( # me: again, step + 1 is an upper bound so fine.
                tokens_l2r[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens_r2l[:, : step + 1] = torch.index_select( # me: again, step + 1 is an upper bound so fine.
                tokens_r2l[:, : step + 1], dim=0, index=active_bbsz_idx
            )

            final_directions_l2r[:, : step + 1] = torch.index_select( # me: again, step + 1 is an upper bound so fine.
                final_directions_l2r[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            final_directions_r2l[:, : step + 1] = torch.index_select( # me: again, step + 1 is an upper bound so fine.
                final_directions_r2l[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            next_l2r_idxes = (tokens_l2r != self.pad).sum(dim=1)
            next_r2l_idxes = (tokens_r2l != self.pad).sum(dim=1)

            # this is really tricky. use SCATTER, the opposite of GATHER, to fill idxes specified by next_l2r_idxes
            tokens_l2r.view(bsz, beam_size, -1).scatter_(
                          dim=2,
                          index=next_l2r_idxes.view(bsz, beam_size).unsqueeze(2),
                          src=torch.gather(cand_indices_l2r, dim=1, index=active_hypos).unsqueeze(2))
            tokens_r2l.view(bsz, beam_size, -1).scatter_(
                          dim=2,
                          index=next_r2l_idxes.view(bsz, beam_size).unsqueeze(2),
                          src=torch.gather(cand_indices_r2l, dim=1, index=active_hypos).unsqueeze(2))
            lasting_directions = torch.gather(directions, dim=1, index=active_hypos) # same gather as above
            assert torch.all(lasting_directions != 0) # don't think any of these sh'b' joining
            directions_l2r_filler = torch.zeros_like(lasting_directions, device=tokens_l2r.device)
            directions_r2l_filler = torch.zeros_like(lasting_directions, device=tokens_l2r.device)
            directions_l2r_filler[lasting_directions == 1] = step+1
            directions_r2l_filler[lasting_directions == 2] = step+1
            final_directions_l2r.view(bsz, beam_size, -1).scatter_(
                          dim=2,
                          index=(next_l2r_idxes-1).view(bsz, beam_size).unsqueeze(2),
                          src=directions_l2r_filler.unsqueeze(2))
            final_directions_r2l.view(bsz, beam_size, -1).scatter_(
                          dim=2,
                          index=(next_r2l_idxes-1).view(bsz, beam_size).unsqueeze(2),
                          src=directions_r2l_filler.unsqueeze(2))
            cols_to_keep_l2r = ((tokens_l2r != self.pad).sum(dim=0) > 0)
            cols_to_keep_r2l = ((tokens_r2l != self.pad).sum(dim=0) > 0)
            tokens_l2r, tokens_r2l = tokens_l2r[:,cols_to_keep_l2r], tokens_r2l[:, cols_to_keep_r2l]
            cols_to_keep_l2r = ((final_directions_l2r != 0).sum(dim=0) > 0)
            cols_to_keep_r2l = ((final_directions_r2l != 0).sum(dim=0) > 0)
            final_directions_l2r, final_directions_r2l = final_directions_l2r[:,cols_to_keep_l2r], final_directions_r2l[:, cols_to_keep_r2l]

            assert final_directions_l2r.shape[0] == final_directions_r2l.shape[0] and tokens_l2r.shape[0] == tokens_r2l.shape[0] and tokens_l2r.shape[0] == final_directions_l2r.shape[0]
            assert final_directions_l2r.shape[1] + 1 == tokens_l2r.shape[1] and final_directions_r2l.shape[1] + 1 == tokens_r2l.shape[1]
            # assert final_directions_l2r.shape == tokens_l2r.shape and final_directions_r2l.shape == tokens_r2l.shape
            assert torch.all((final_directions_l2r[:,:] != 0) == (tokens_l2r[:,1:] != self.pad))
            assert torch.all((final_directions_r2l[:,:] != 0) == (tokens_r2l[:,1:] != self.pad))

            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            assert attn is None, 'deleted this section'

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        do_rerank = self.model.model.task.rerank_test
        if do_rerank: # should only plan to do for test set, since not sure how warmup/order being on/off at various points of training affects it.
            self.runMmlReranker(net_input, finalized, bos_token)

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            if 'old_score' in finalized[0][0]:
                old_scores = torch.tensor([float(elem["old_score"].item()) for elem in finalized[sent]])
                _, sorted_old_scores_indices = torch.sort(old_scores, descending=True)
                lem = ''.join([self.tgt_dict[q] for q in sample['net_input']['src_tokens'][sent]])
                old = ''.join([self.tgt_dict[q] for q in finalized[sent][sorted_old_scores_indices[0]]['tokens']][:-1])
                new = ''.join([self.tgt_dict[q] for q in finalized[sent][sorted_scores_indices[0]]['tokens']][:-1])
                # if sorted_scores_indices[0].item() != sorted_old_scores_indices[0].item():
                #     print("LEM:", lem)
                #     print("OLD:", old)
                #     print("NEW:", new)
                #     print("COR:", ''.join([self.tgt_dict[q] for q in sample['target'][sent][sample['target'][sent] != 1]]))
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def runMmlReranker(self, net_input, finalized, bos_token):
        model_mml_reranker = self.model
        assert model_mml_reranker.model.tp == 'fullbidi'
        beam_size =self.beam_size# len(finalized[0])
        if set([len(finalized[i]) for i in range(len(finalized))]) != {beam_size}:
            # EXTREMELy rare quirk where occasionally an example that we are reranking has fewer than beam_size examples
            for i in range(len(finalized)):
                if len(finalized[i]) != beam_size:
                    try:
                        assert len(finalized[i]) < beam_size, str(len(finalized[i])) + ' ' + str(beam_size)
                    except:
                        print(finalized[i])
                        print(beam_size)
                        assert len(finalized[i]) < beam_size, str(len(finalized[i])) + ' ' + str(beam_size) # so it still fails
                    while len(finalized[i]) < beam_size: finalized[i].append(finalized[i][0]) # makes no difference
        bsz = net_input['src_tokens'].shape[0]
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(net_input['src_tokens'].device).long()

        src_tokens, src_lengths = net_input['src_tokens'][new_order,:], net_input['src_lengths'][new_order]

        # Get decoder ins
        inputs_l2r = [x['tokens'] for i in range(len(finalized)) for x in finalized[i]]
        lengths = [x.shape[0] for x in inputs_l2r]
        max_len = max(lengths)
        tokens_l2r, trg_l2r = [None] * len(inputs_l2r),[None] * len(inputs_l2r)
        for i in range(len(inputs_l2r)):
            x = inputs_l2r[i]
            if x.shape[0] < max_len:
                trgi = torch.cat([x, torch.full((max_len-x.shape[0],), self.pad, device=net_input['src_tokens'].device)])
            else: trgi = x
            y = torch.cat([torch.full((1,), self.eos if bos_token is None else bos_token, device=net_input['src_tokens'].device), x[:-1]])
            if y.shape[0] < max_len:
                srci = torch.cat([y, torch.full((max_len-y.shape[0],), self.pad, device=net_input['src_tokens'].device)])
            else: srci = y
            tokens_l2r[i], trg_l2r[i] = srci, trgi

        inputs_l2r = torch.stack(tokens_l2r)
        trg_l2r = torch.stack(trg_l2r)
        inputs_r2l, trg_r2l = reverseSequence(inputs_l2r, trg_l2r)

        # Now keep going sort of like in bidi_loss
        lengths = torch.sum(inputs_l2r != self.pad, dim=1) + 1

        assert torch.all(torch.gather(trg_l2r, 1,index=((trg_l2r != self.pad).sum(dim=1) - 1).unsqueeze(1)) == self.eos) and \
               torch.all(torch.gather(trg_r2l, 1,index=((trg_r2l != self.pad).sum(dim=1) - 1).unsqueeze(1)) == self.eos)

        gath = [createGatherers(length.item()) for length in lengths]
        gath0 = [q[0] for q in gath]
        gath_concat = torch.cat(gath0, dim=0)

        tok_l2r, tok_r2l, join_probs, order_probs = model_mml_reranker.model(src_tokens.long(), src_lengths, inputs_l2r.long(), inputs_r2l.long(),
                                                          self.tgt_dict,
                                                          do_rand=False)

        assert len(join_probs.shape) == 2, "for training, shoyuld only output 2 things"

        gold_join = torch.cat([x[1] for x in gath], dim=0).to(tok_l2r.device)

        assert tok_l2r.shape[0] == join_probs.shape[0] and tok_r2l.shape[0] == join_probs.shape[0]

        gath0 = [g[0] for g in gath]
        gath_concat = torch.cat(gath0, dim=0)
        gath_wd_idxes = torch.cat(
            [torch.full((len(gath0[i]),), i, device=tok_l2r.device) for i in range(len(gath0))])

        probs_l2r = model_mml_reranker.model.get_normalized_probs(tok_l2r, log_probs=True)  # takes log softmax xs last dim, so fine
        probs_r2l = model_mml_reranker.model.get_normalized_probs(tok_r2l, log_probs=True)  # takes log softmax xs last dim, so fine
        probs_join = model_mml_reranker.model.get_normalized_probs(join_probs, log_probs=True)  # takes log softmax xs last dim, so fine

        if order_probs is None:
            order_probs = torch.zeros_like(join_probs).fill_(math.log(0.5))
            probs_order = order_probs
        else:
            probs_order = model_mml_reranker.model.get_normalized_probs(order_probs, log_probs=True)

        scores = compute_advanced_loss(probs_l2r, trg_l2r, probs_r2l, trg_r2l, probs_join, gold_join, probs_order,
                              gath_concat, gath_wd_idxes, True, True)

        if self.normalize_scores:
            lengths = torch.tensor(lengths)
            penalties = lengths ** (self.len_penalty)
            scores = scores / penalties
        scores = scores.reshape(bsz, beam_size)
        for i in range(len(finalized)):
            for j in range(beam_size):
                finalized[i][j]['old_score'] = finalized[i][j]['score']
                finalized[i][j]['score'] = scores.reshape(bsz, beam_size)[i,j]


    def _prefix_tokens(
            self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.min(prefix_lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                         :, 0, 1: step + 1
                         ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
            self,
            step: int,
            bbsz_idx,
            eos_scores,
            tokens_l2r,
            tokens_r2l,
            final_directions_l2r,
            final_directions_r2l,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            attn: Optional[Tensor],
            src_lengths,
            max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone_l2r = tokens_l2r.index_select(0, bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS
        tokens_clone_r2l = tokens_r2l.index_select(0, bbsz_idx)[
                       :, 0: step + 2 # ME: CHANGED TO 0, b/c once reversed it's basically the eos added
                       ]  # skip the first index, which is EOS
        cols_to_keep_l2r = ((tokens_clone_l2r != self.pad).sum(dim=0) > 0)
        cols_to_keep_r2l = ((tokens_clone_r2l != self.pad).sum(dim=0) > 0)
        tokens_clone_l2r, tokens_clone_r2l = tokens_clone_l2r[:,cols_to_keep_l2r], tokens_clone_r2l[:,cols_to_keep_r2l]
        tokens_clone = torch.full((bbsz_idx.numel(), step+1), 1, device=tokens_clone_l2r.device)
        tokens_clone[:,:tokens_clone_l2r.shape[1]] = tokens_clone_l2r
        reversed= tokens_clone_r2l[:,torch.arange(tokens_clone_r2l.shape[1] - 1, -1, -1)]
        tokens_clone.view(-1)[tokens_clone.view(-1) == 1] = reversed.view(-1)[reversed.view(-1) != 1]
        assert torch.all(tokens_clone != 1) and torch.all(tokens_clone[:,-1] == self.eos)

        # now do directions
        directions_l2r = final_directions_l2r.index_select(0, bbsz_idx)
        directions_r2l = final_directions_r2l.index_select(0, bbsz_idx)
        dcols_to_keep_l2r = ((directions_l2r != 0).sum(dim=0) > 0)
        dcols_to_keep_r2l = ((directions_r2l != 0).sum(dim=0) > 0)
        directions_l2r, directions_r2l = directions_l2r[:,dcols_to_keep_l2r], directions_r2l[:,dcols_to_keep_r2l]
        directions = torch.zeros(bbsz_idx.numel(), step+1, device=directions_l2r.device).long()
        directions[:,:directions_l2r.shape[1]] = directions_l2r
        dreversed= directions_r2l[:,torch.arange(directions_r2l.shape[1] - 1, -1, -1, device=directions_r2l.device)]
        directions[:, :-1][directions[:, :-1] == 0] = dreversed[dreversed != 0]
        if dreversed.shape[1] > 0: # this is needed in case everything is from left
            directions[dreversed[:,0] == step,-1] = -2 # last dir was right
            directions[dreversed[:,0] != step,-1] = -1 # last dir was left
        else: directions[:,-1] = -1
        assert torch.all(directions != 0)

        attn_clone = None

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = bbsz_idx // beam_size
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                add_in=  True
                for ppp in range(len(finalized[sent_list[i]])):
                    qqq = finalized[sent_list[i]][ppp]
                    tokens_h = qqq['tokens']
                    tci = tokens_clone[i]
                    if tci.shape == tokens_h.shape and torch.all(tci == tokens_h):
                        if qqq['score'] >= eos_scores[i]:
                            add_in = False
                            break
                        else:
                            finalized[sent_list[i]][ppp] = {
                            "tokens": tokens_clone[i],
                            "score": eos_scores[i],
                            "attention": hypo_attn,  # src_len x tgt_len
                            "alignment": torch.empty(0),
                            "positional_scores": pos_scores[i],
                            "directions": directions[i]
                            }
                            add_in=False
                            break
                if add_in:
                    finalized[sent_list[i]].append(
                        {
                            "tokens": tokens_clone[i],
                            "score": eos_scores[i],
                            "attention": hypo_attn,  # src_len x tgt_len
                            "alignment": torch.empty(0),
                            "positional_scores": pos_scores[i],
                            "directions": directions[i]
                        }
                    )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                    step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unfin_idx: int,
            max_len: int,
            finalized_sent_len: int,
            beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

