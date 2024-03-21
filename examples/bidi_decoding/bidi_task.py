import datetime
import json
from argparse import Namespace

import logging
from typing import Optional
from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.data import (
    LanguagePairDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
logger = logging.getLogger(__name__)

from fairseq.models import (
    register_model_architecture,
    register_model
)
from fairseq.data import Dictionary


from fairseq.models.transformer import (
    base_architecture,
    TransformerModel
)
import torch

try:
    from examples.bidi_decoding.full_bidi_transformer import DummyEncoder
except:
    from bidi_decoding.full_bidi_transformer import DummyEncoder


@dataclass
class BidiTaskConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to data directory"
        },
    )

    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )

    do_r2l: bool = field(
        default=False, metadata={"help": "do r2l if unidirectional"}
    )

    need_extra_spec: bool = field(
        default=False,
        metadata={
            "help": 'for bidi'
        },
    )

    rerank_test: bool = field(
        default=False,
        metadata={
            "help": 'do reranking under mml'
        },
    )

    force_l2r: bool = field(
        default=False,
        metadata={
            "help": 'force l2r decoding'
        },
    )

    force_r2l: bool = field(
        default=False,
        metadata={
            "help": 'force r2l decoding'
        },
    )

    rerank_sepish: Optional[str] = field( # just-eval-test-file
        default="",
        metadata={
            "help": 'id for sep / sep_rand model you wanna rearank with mml'
        },
    )
    rerank_l2r: Optional[str] = field( # just-eval-test-file
        default="",
        metadata={
            "help": 'id for an l2r model you want to use as input'
        },
    )
    rerank_r2l: Optional[str] = field(  # just-eval-test-file
        default="",
        metadata={
            "help": 'id for an r2l model you want to use as input'
        },
    )

    do_eval: bool = field(
        default=False, metadata={"help": "Do evaluation"}
    )

    generate_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for evaluation, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )

    just_eval_test_file: Optional[str] = field( # just-eval-test-file
        default="",
        metadata={
            "help": 'ID for a run you want to produce predictions for; no training at all'
        },
    )

    cont_curr: bool = field(
        default=False,
        metadata={
            "help": 'whether to continue the run'
        },
    )


class BidiTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    def __init__(self, cfg: BidiTaskConfig, src_dict, tgt_dict, data_train, data_val, data_test):
        super().__init__(cfg)
        self.just_eval_test_file = cfg.just_eval_test_file
        self.rerank_test = cfg.rerank_test
        self.rerank_sepish = cfg.rerank_sepish
        self.rerank_l2r = cfg.rerank_l2r
        self.rerank_r2l = cfg.rerank_r2l
        self.need_extra_spec = cfg.need_extra_spec
        self.force_l2r = cfg.force_l2r
        self.force_r2l = cfg.force_r2l
        self.cont_curr = cfg.cont_curr

        if self.rerank_l2r != '':
            assert self.rerank_test
            assert self.rerank_r2l == ''
            assert self.rerank_sepish == ''
            assert self.just_eval_test_file != ''

        if self.rerank_r2l != '':
            assert self.rerank_test
            assert self.rerank_l2r == ''
            assert self.rerank_sepish == ''
            assert self.just_eval_test_file != ''

        if self.rerank_sepish != '':
            assert self.rerank_test, 'should both be selected'
            assert self.just_eval_test_file != ''
            assert self.rerank_l2r == ''
            assert self.rerank_r2l == ''

        if self.rerank_test:
            assert self.just_eval_test_file != ''

        if self.force_l2r: assert not self.force_r2l
        if self.force_r2l: assert not self.force_l2r
        if self.force_r2l or self.force_l2r:
            assert not self.rerank_test
            assert self.just_eval_test_file != ''

        self.rerank_dictionary = None # for rerank

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_train = data_train
        self.data_val = data_val
        self.data_test, self.data_test_rerank = data_test, None
        if type(data_test) == tuple:
            self.data_test = data_test[0]
            self.data_test_rerank = data_test[1]

        self.best_metrics = {} # dict to store this

        self.is_evaluating_test = False # will flip to true (externally) at very end, when edvaluating test
        self.write_to_file_name = None # another things wedged in here
        self.evaluate_immediately = False # hack
        self.keep_training_from_best = False # hack

    @classmethod
    def setup_task(cls, cfg: BidiTaskConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        DO_TEST=True

        assert len(utils.split_paths(cfg.data)) == 1

        if cfg.rerank_test:
            if cfg.rerank_l2r:
                file_to_read, style = cfg.rerank_l2r, 'l2r'
            elif cfg.rerank_r2l:
                file_to_read, style = cfg.rerank_r2l, 'r2l'
            elif cfg.rerank_sepish:
                file_to_read, style = cfg.rerank_sepish, 'bidi'
            else:
                file_to_read, style = cfg.just_eval_test_file, 'bidi'
        else: file_to_read, style = None, None

        extra = [] if not cfg.need_extra_spec else ['<sep2>', '<cls_j>', '<cls_o>']
        data_train, data_val, data_test, d_src, d_trg = cls.getDataset(cfg,
                                                                       extra=extra,
                                                                       do_test = DO_TEST)

        if cfg.do_r2l:
            if cfg._name != 'inflection': assert False, 'think this assumes certain data format, so fix that.'
            assert file_to_read is None and style is None
            for x in data_train:
                x[1] = x[1][::-1]
            for x in data_val:
                x[1] = x[1][::-1]
            if DO_TEST:
                for x in data_test:
                    x[1] = x[1][::-1]

        if file_to_read is not None:
            assert DO_TEST
            assert not cfg.do_r2l, 'should always be bidi model, even if reading in results from a different style'
            data_test, data_to_rerank = cls.readFileForRerank('checkpoints/' + file_to_read + '/outputs.txt', style, data_test)
            data_test = (data_test, data_to_rerank)

        # load dictionaries
        src_dict = d_src
        tgt_dict = d_trg
        if src_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        logger.info("dictionary: {} types".format(len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict, data_train, data_val, data_test)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def getBestMetric(self, name, val, do_min=True):
        prev_best = self.best_metrics.get(name, 9999999999 if do_min else -9999999999)
        if val < prev_best and do_min:
            self.best_metrics[name] = val
        elif val > prev_best and not do_min:
            self.best_metrics[name] = val
        return self.best_metrics[name]


    def build_model(self, cfg, do_generator=True):
        decoder_only = self.src_dict is None
        self.src_dict = self.tgt_dict # for now initialize it with src, then cut it out.
        model = super().build_model(cfg)
        if decoder_only:
            self.src_dict = None
            model.encoder = DummyEncoder() # No encoder
        model.task = self
        if self.cfg.do_eval and do_generator:
            gen_args = json.loads(self.cfg.generate_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.do_eval:
            total, stats, extra = self.inference(self.sequence_generator, sample, model)
            logging_output["_total"] = total
            if extra is not None:
                for k, v in extra.items(): logging_output[k] = v
            if stats is not None:
                num_fully_left, num_fully_right, num_neither, num_left_moves, num_right_moves = stats
                logging_output["_num_full_l2r"] = num_fully_left
                logging_output["_num_full_r2l"] = num_fully_right
                logging_output["_num_neither"] = num_neither
                logging_output["_num_left_moves"] = num_left_moves
                logging_output["_num_right_moves"] = num_right_moves
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.do_eval:
            if len(logging_outputs) > 1:
                assert '_total' not in logging_outputs[0] # should be training
            else: assert len(logging_outputs) == 1
            if '_total' in logging_outputs[0]:
                metrics.log_scalar("_total", logging_outputs[0]['_total'])
                if '_num_full_l2r' in logging_outputs[0]:
                    metrics.log_scalar("_num_full_l2r", logging_outputs[0]['_num_full_l2r'])
                    metrics.log_scalar("_num_full_r2l", logging_outputs[0]['_num_full_r2l'])
                    metrics.log_scalar("_num_neither", logging_outputs[0]['_num_neither'])
                    metrics.log_scalar("_num_left_moves", logging_outputs[0]['_num_left_moves'])
                    metrics.log_scalar("_num_right_moves", logging_outputs[0]['_num_right_moves'])

                def perc_full_l2r(meters):
                    acc = meters['_num_full_l2r'].sum / meters['_total'].sum * 100
                    return acc.item() if type(acc) == torch.Tensor else acc

                def perc_full_r2l(meters):
                    acc = meters['_num_full_r2l'].sum / meters['_total'].sum * 100
                    return acc.item() if type(acc) == torch.Tensor else acc

                def perc_neither(meters):
                    acc = meters['_num_neither'].sum / meters['_total'].sum * 100
                    return acc.item() if type(acc) == torch.Tensor else acc

                def avg_perc_left_moves(meters):
                    acc = meters['_num_left_moves'].sum / meters['_total'].sum * 100
                    return acc.item() if type(acc) == torch.Tensor else acc

                def avg_perc_right_moves(meters):
                    acc = meters['_num_right_moves'].sum / meters['_total'].sum * 100
                    return acc.item() if type(acc) == torch.Tensor else acc

                if '_num_full_l2r' in logging_outputs[0]:
                    metrics.log_derived("val_perc_full_l2r", perc_full_l2r)
                    metrics.log_derived("val_perc_full_r2l", perc_full_r2l)
                    metrics.log_derived("val_perc_neither", perc_neither)
                    metrics.log_derived("val_avg_perc_left_moves", avg_perc_left_moves)
                    metrics.log_derived("val_avg_perc_right_moves", avg_perc_right_moves)

                self.log_additional_metrics(logging_outputs, metrics)

    def call_orig_reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def log_additional_metrics(self, logging_outputs, metrics):
        return


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        src_sents, src_lengths = [], []
        trg_sents, trg_lengths = [], []
        if split == 'train': data = self.data_train
        elif split == 'valid': data = self.data_val
        else:
            assert split == 'test'
            if self.data_test_rerank is None: data = self.data_test
            else:
                assert len(self.data_test) == len(self.data_test_rerank)
                new_data = []
                for i in range(len(self.data_test)):
                    orig, to_rerank = self.data_test[i], self.data_test_rerank[i]
                    ex = self.extract_example_from_rerank(orig, to_rerank)
                    new_data.append(ex)
                data = new_data
                self.rerank_dictionary = []

        for ex in data:
            src_ex = self.tokenize_src(ex)
            trg_ex = self.tokenize_trg(ex)
            assert trg_ex[0] not in {self.tgt_dict.eos(), self.tgt_dict.bos()} #move_eos_to_beginning in language_pair_dataset.py is set to True, so shouldn't start with that.

            if self.rerank_test and split == 'test':
                trgs = self.tokenize_trgs_for_rerank(ex)
                for x in trgs: assert x[0] != self.tgt_dict.eos() #or x[0] == self.tgt_dict.eos() and len(x) == 1 # latter case in case epsilon predicted...not sure if should allow or if something got changed to make that happen since it wasn't before i don't think.....
                self.rerank_dictionary.append((src_ex, trgs))

            if src_ex is not None:
                src_lengths += [len(src_ex)]
                src_sents += [torch.IntTensor(src_ex).to(torch.int64)]
            else: src_lengths, src_sents = None, None

            trg_lengths += [len(trg_ex)]
            trg_sents += [torch.IntTensor(trg_ex).to(torch.int64)]

        eos = self.src_dict.eos() if src_lengths is not None else self.tgt_dict.eos()

        if src_lengths is None:
            # Make dummy src sentences that will be ignored in the model (in fact, whoever's calling the model won't deal with it)
            src_lengths = [1] * len(trg_lengths)
            src_sents = [torch.IntTensor([eos]).to(torch.int64)] * len(trg_sents)
            src_dict = self.tgt_dict
        else: src_dict = self.src_dict


        self.datasets[split] = LanguagePairDataset(
            src_sents,
            src_lengths,
            src_dict,
            trg_sents,
            trg_lengths,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            # align_dataset=align_dataset,
            eos=eos,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=split != 'test',
            # pad_to_multiple=self.cfg.required_seq_len_multiple,
        )


    @classmethod
    def getDataset(cls,
                   cfg,
                   extra=[],
                   do_test=True):
        raise NotImplementedError("Must implement in subclass.")

    @classmethod
    def readFileForRerank(cls, checkpoint, style, real):
        raise NotImplementedError("Must implement in subclass.")

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        raise NotImplementedError("Not implemented yet - do not implement in subclass. Use BidiTask arguments to perform inference.")

    def inference(self, generator, sample, model):
        raise NotImplementedError("Must implement in subclass.")

    def tokenize_src(self, example):
        raise NotImplementedError("Must implement in subclass.")

    def tokenize_trg(self, example):
        raise NotImplementedError("Must implement in subclass.")

    def tokenize_trgs_for_rerank(self, example):
        raise NotImplementedError("Must implement in subclass.")

    def extract_example_from_rerank(self, orig, to_rerank):
        raise NotImplementedError("Must implement in subclass.")


def buildVocab(data, extra = [], share_dict = True):
    d_src = Dictionary(extra_special_symbols=['<sep>'] + extra)
    if not share_dict: d_trg = Dictionary(extra_special_symbols=['<sep>'] + extra)
    for x in data:
        assert type(x) == tuple or type(x) == str
        src = x[0] if type(x) == tuple else x
        for char in src: # source
            assert '<' not in char, 'reserving this for special symbols'
            assert '¢' not in char, 'used as special symbol in writing output files'
            d_src.add_symbol(char)
        if type(x) == tuple:
            for char in x[1]: # target
                assert '<' not in char, 'reserving this for special symbols'
                assert '¢' not in char, 'used as special symbol in writing output files'
                if share_dict: d_src.add_symbol(char)
                else: d_trg.add_symbol(char)
    d_src.finalize(padding_factor=0)
    if not share_dict: d_trg.finalize(padding_factor=0)
    else: d_trg = d_src
    return (d_src, d_trg)


@register_model("decoder_only_transformer")
class DecoderOnlyTransformer(TransformerModel):
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=None,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out



@register_model_architecture('transformer', 'bidi_transformer')
def bidi_transformer(args): # Note this gets called twice
    if args.criterion == 'bidi_loss':
        assert args.loss_style in {'sep', 'sep-rand', 'MML'} and args.need_extra_spec, 'if doing bidirectional, need to have --need-extra-spec in the cmd line arguments'
        args.do_order = args.loss_style == 'MML'

    if args.do_r2l: assert args.criterion == 'cross_entropy'
    if args.criterion != 'bidi_loss': args.label_smoothing = 0.1
    args.save_dir = args.save_dir + ("/" if args.save_dir[-1] != '/' else "") + 'Run_'+ datetime.datetime.utcnow().strftime("%m-%d-%Y_%H-%M-%S").replace('-','_')

    return base_architecture(args)
