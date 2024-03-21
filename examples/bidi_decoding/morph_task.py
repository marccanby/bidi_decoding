import datetime
import json
import pandas as pd
from argparse import Namespace
import random

import logging
from typing import Optional
from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.data import (
    LanguagePairDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
logger = logging.getLogger(__name__)

import torch
import ast

from fairseq.models import (
    register_model_architecture
)

try:
    from examples.bidi_decoding.bidi_task import BidiTaskConfig, BidiTask, buildVocab, bidi_transformer
except:
    from bidi_decoding.bidi_task import BidiTaskConfig, BidiTask, buildVocab, bidi_transformer

try:
    from examples.bidi_decoding.bidi_utils import getOrderingStats, getDirections
except:
    from bidi_decoding.bidi_utils import getOrderingStats, getDirections



@dataclass
class InflectionConfig(BidiTaskConfig):
    lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        }
    )

    new_schema: bool = field(
        default=False,
        metadata={
            "help": 'wther or not to use new unimorph schema'
        },
    )

    hyper_size: Optional[str] = field(
        default=None,
        metadata={'help': 'Avoid having to specify individual hyperparameters. Should be in {small, medium, large}'})

    eval_acc_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("inflection", dataclass=InflectionConfig)
class InflectionTask(BidiTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    def __init__(self, cfg: InflectionConfig, src_dict, tgt_dict, data_train, data_val, data_test):
        super().__init__(cfg, src_dict, tgt_dict, data_train, data_val, data_test)
        self.new_schema = cfg.new_schema

    def extract_example_from_rerank(self, orig, to_rerank):
        assert orig[0] == to_rerank[0]
        assert orig[1] == to_rerank[1]
        assert orig[2] == to_rerank[2]
        assert len(to_rerank) in {7, 9}  # 7 = l2r/r2l, 9 = bidi
        if len(to_rerank) == 7:
            ex = [orig[0], orig[1], orig[2]] + to_rerank[5]
        else:
            ex = [orig[0], orig[1], orig[2]] + to_rerank[6]

        return ex

    def tokenize_src(self, example):
        src_ex = [self.src_dict.eos()]
        for char in example[0]:
            src_ex += [self.src_dict.index(char)]
        src_ex += [self.src_dict.index('<sep>')] # Appears I use this both to separate lemma and tags and as cls token for left side.
        if not self.new_schema:
            spltherehere = example[2].split(';')
        else:
            spltherehere = tokenize_tags_new(example[2])
        for tag in spltherehere:
            src_ex += [self.src_dict.index(tag)]
        src_ex += [self.src_dict.eos()]
        return src_ex

    def tokenize_trg(self, example):
        return tokenize_trg_word(example[1], self.tgt_dict)

    def tokenize_trgs_for_rerank(self, example):
        res = []
        for wd in example[3:]:
            if wd == '':
                assert False, 'dont support reranking empty strings'
            res += [tokenize_trg_word(wd, self.tgt_dict)]
        return res

    def log_additional_metrics(self, logging_outputs, metrics):
        # We use accuracy for this task, so we will log the total correct.

        metrics.log_scalar("_correct", logging_outputs[0]['_correct'])

        def accuracy(meters):
            acc = meters['_correct'].sum / meters['_total'].sum
            return acc.item() if type(acc) == torch.Tensor else acc

        metrics.log_derived("val_acc", accuracy)

    def decode(self, toks, src=None):
        lst_to_join = []
        unk_is_replaced=''
        for idx in toks.int().cpu():
            if '<' not in self.tgt_dict.symbols[idx]:
                lst_to_join.append(self.tgt_dict.symbols[idx])
            elif idx == self.tgt_dict.unk_index and src is not None:
                unk_char = None
                for ch in src:
                    if self.tgt_dict.unk_index == self.tgt_dict.index(ch):
                        unk_char = ch
                        break
                if unk_char is not None:
                    lst_to_join.append(unk_char)
                    unk_is_replaced =unk_is_replaced+ 'UNK REPLACED: ' + unk_char + '; '
        s = "".join(lst_to_join)
        if unk_is_replaced != '':
            unk_is_replaced = unk_is_replaced + src + ' ' + s
            print (unk_is_replaced)
        return s

    def inference(self, generator, sample, model):

        doing_val = not self.is_evaluating_test
        data = self.data_val if doing_val else self.data_test
        data = [list(x) for x in data]

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)

        lemmas, tags, refs = [], [], [] # For the inputs (srcs) and gold trgs
        hyps, dirs = [], [] # For the best output
        hyps_all, scores_all, dirs_all = [], [], [] # For all the outputs of beam search
        for i in range(len(gen_out)):

            # For the inputs (srcs) and gold trgs
            lemmas.append(data[sample['id'][i]][0])
            tags.append(data[sample['id'][i]][2])
            refs.append(self.decode(utils.strip_pad(sample["target"][i], self.tgt_dict.pad())))

            # For the best output
            hyps.append(self.decode(gen_out[i][0]["tokens"], data[sample['id'][i]][0]))

            # For all the outputs
            options = [self.decode(gen_out[i][jj]["tokens"], data[sample['id'][i]][0]) for jj in range(len(gen_out[i]))]
            scores = [str(gen_out[i][jj]["score"].item()) for jj in range(len(gen_out[i]))]

            # Replace ;'s with ¢ in output sentences, so ;'s can be used to separate hypotheses in output files
            for jjj in range(len(options)):
                assert '¢' not in options[jjj]
                options[jjj] = options[jjj].replace(';', '¢')

            # Save all the outputs of beam search
            hyps_all.append(';'.join(options))
            scores_all.append(';'.join(scores))

            # Compute directions
            if 'directions' in gen_out[i][0]:
                directions, dir_string = getDirections(gen_out[i])
                dirs.append(directions)
                dirs_all.append(dir_string)

            # Case if reference output has UNK tokens (only way to get this right would be to resolve UNK's in smart way)
            if refs[-1] != data[sample['id'][i]][1]:
                assert self.tgt_dict.unk_index in sample["target"][i]
                print("UNK in ref!! "+ refs[-1] + ' ' + data[sample['id'][i]][1] + ' ' + options[0])
                refs[-1] = data[sample['id'][i]][1] # Save string before replacing with UNK's because this is the true gold.

        # Sanity check
        if len(dirs) > 0: assert len(dirs) == len(lemmas) and len(dirs_all) == len(lemmas)
        assert len(hyps_all) == len(scores_all) and len(hyps_all ) == len(lemmas) and len(hyps) == len(refs)

        # Do printing
        if self.cfg.eval_acc_print_samples:
            idx = random.randint(0, len(lemmas)-1)
            logger.info("Example lemma: " + lemmas[idx])
            logger.info("Example reference: " + refs[idx])
            logger.info("Example hypothesis: " + hyps[idx])
            if len(dirs) > 0: logger.info("Example direction: " + str(dirs[idx]))

        if len(dirs) > 0:
            stats = getOrderingStats(dirs)
        else: stats = None

        # Save to file - should be consistent with order in the readFileForRerank function
        if self.write_to_file_name is not None:
            lines = [lemmas[i] + '\t' +
                     tags[i] + '\t' + refs[i] +
                     '\t' + hyps[i] +
                     (('\t' + str(dirs[i])) if len(dirs) > 0 else "") +
                     '\t' + hyps_all[i] + '\t' +
                     scores_all[i] +
                     (('\t' + str(dirs_all[i])) if len(dirs) > 0 else "") for i in range(len(lemmas))]
            with open(self.write_to_file_name, 'a') as fil:
                for l in lines:
                    fil.write(l + '\n')

        # Compute inference metrics and return
        correct = sum([hyps[i] == refs[i] for i in range(len(refs))])
        total = len(refs)
        return total, stats, {'_correct': correct}

    @classmethod
    def getDataset(cls, cfg, extra=[],  do_test=True):
        paths = utils.split_paths(cfg.data)
        assert len(paths) == 1
        path = paths[0]
        lang, new_schema = cfg.lang, cfg.new_schema

        data_train, data_val, data_test = getTrainValTestDataInflection(path, lang, do_test=do_test, new_schema=new_schema)

        # Because inflection task has lemma + tags in input, got to concatenate these so they both end up in the vocabulary.
        # General tasks won't need this.
        train_input_for_vocab = [None] * len(data_train)
        tagset = set()
        for i in range(len(data_train)):
            x = data_train[i]
            splthere = x[2].split(';') if not new_schema else tokenize_tags_new(x[2])
            for tag in splthere: tagset.add(tag)
            inp = list(x[0]) + splthere
            out = list(x[1])
            train_input_for_vocab[i] = (inp, out)

        (d_src, d_trg) = buildVocab(train_input_for_vocab, extra = extra)
        d_src.tagset = tagset
        d_trg.tagset = tagset

        return data_train, data_val, data_test, d_src, d_trg

    @classmethod
    def readFileForRerank(cls, checkpoint, style, real):
        assert style in {'l2r', 'r2l', 'bidi'}
        # Remove dups removes exact dups (lemma, tags, form) that are in the ORIGINAL TEST SET
        # If False, it preserves it as it was originally, consistent with the #'s reported by the fairseq code at the moment
        filename = checkpoint
        df = pd.read_csv(filename, sep='\t', header=None).values.tolist()
        has_order = style == 'bidi'
        if not has_order:
            if style == 'r2l':
                df = [(x[0], x[1], x[2][::-1] if str(x[2]) != 'nan' else 'nan', x[3][::-1] if str(x[3]) != 'nan' else 'nan', ';'.join([y[::-1] for y in x[4].split(';')]), x[5]) for x in df]
            df = [(x[0], x[1], x[2], x[2] == (x[3] if str(x[3]) != 'nan' else 'nan'), (x[3] if str(x[3]) != 'nan' else 'nan') + '\t' + x[4] + '\t' + x[5]) for x in df]
        else:
            df = [(x[0], x[1], x[2], x[3] if type(x[3]) == str else '', x[4], x[5], x[6], x[7]) for x in df] # not sure why x[3] isn't always a string! think it perhaps has to do with being empty, maybe if a special symbol was predicted
            df = [(x[0], x[1], x[2], x[2] == (x[3] if str(x[3]) != 'nan' else 'nan'), (x[3] if str(x[3]) != 'nan' else 'nan') + '\t' + x[4] + '\t' + x[5] + '\t' + x[6] + '\t' + x[7]) for x in
                  df]

        # Read in real set, and figure out which examples are dups
        real = [tuple(q) for q in real]
        dups = {x: real.count(x) for x in real if real.count(x) > 1}  # Where entire line is duped in the test set

        # Sanity assertion - make sure model always returns same string for particular input (lemma, tag)
        set_of_lemma_tag_forms = {(x[0], x[1], x[2]): {} for x in df}
        for x in df:
            set_of_lemma_tag_forms[(x[0], x[1], x[2])][x[3]] = set_of_lemma_tag_forms[(x[0], x[1], x[2])].get(x[3], 0)
            set_of_lemma_tag_forms[(x[0], x[1], x[2])][x[3]] += 1
        to_keep = {}
        for x in set_of_lemma_tag_forms:
            # NOTE: Below in the latter case, can technically be <=, but will cause below code to fail. Works as is for now.
            try:
                assert len(set_of_lemma_tag_forms[x]) == 1 or (x[0], x[2], x[1]) in dups and len(
                    set_of_lemma_tag_forms[x]) == dups[(x[0], x[2], x[1])]
            except:
                if len(df) == len(real): continue  # No problem here, this occurs with the <= case i believe, but if lens are equal dont even run into hot water later
                assert False

        df_new = []
        cnt, cnt_both_same, cnt_form_same, cnt_diff = 0, 0, 0, 0
        for x in df:
            if (x[0], x[1], x[2]) not in to_keep:
                df_new.append(x)
            else:
                cnt += 1
                if x[3] == to_keep[(x[0], x[1], x[2])]:
                    cnt_both_same += 1
                elif x[3].split('\t')[0] == to_keep[(x[0], x[1], x[2])].split('\t')[0]:
                    assert False, 'dont think needed'
                    cnt_form_same += 1
                else:
                    cnt_diff += 1
                df_new.append(
                    (x[0], x[1], x[2], to_keep[(x[0], x[1], x[2])], x[2] == to_keep[(x[0], x[1], x[2])].split('\t')[0]))
        df = df_new


        # Sort both sets
        real = sorted(real, key=lambda x: (x[0], x[2], x[1]))
        df = sorted(df, key=lambda x: (x[0], x[1], x[2]))
        assert real == [(x[0], x[2], x[1]) for x in df], str(len(real)) + ' ' + str(len(df))

        # NOTE: Have reordered these down here for pycharm code (needs to be lemma, ref, tags, etc....)
        # So it's DIFFERNT (slightly) from the other one
        if not has_order:
            df = [(x[0], x[2], x[1], x[3], x[4].split('\t')[0], x[4].split('\t')[1].split(';'), [float(y) for y in x[4].split('\t')[2].split(';')]) for x in df]
            for i in range(len(df)):
                cpy = list(df[i])
                if '¢' in df[i][4]:
                    df[i][4] = df[i][4].replace('¢', ';')
                for j in range(len(df[i][5])):
                    if '¢' in df[i][5][j]: df[i][5][j] = df[i][5][j].replace('¢', ';')
                df[i] = tuple(cpy)
        else:
            df = [(x[0], x[2], x[1], x[3], x[4].split('\t')[0], ast.literal_eval(x[4].split('\t')[1]),
                   x[4].split('\t')[2].split(';'), [float(y) for y in x[4].split('\t')[3].split(';')],
                   [ast.literal_eval(y) for y in x[4].split('\t')[4].split(';')]) for x in df]
            for i in range(len(df)):
                cpy = list(df[i])
                if '¢' in df[i][4]:
                    df[i][4] = df[i][4].replace('¢', ';')
                for j in range(len(df[i][6])):
                    if '¢' in df[i][6][j]: df[i][6][j] = df[i][6][j].replace('¢', ';')
                df[i] = tuple(cpy)

        return real, df


def tokenize_trg_word(word, tgt_dict):
    trg_ex = []  # Making it empty, since this move_eos_to_beginning in language_pair_dataset.py is set to True
    for char in word:
        trg_ex += [tgt_dict.index(char)]
    trg_ex += [tgt_dict.eos()]
    return trg_ex


def readUnimorphData(file, do_print=True):
    if do_print:
        print("Reading file '" + file + "'...")

    # Read the file and split into lines
    try:
        lines = open(file, encoding='utf-8').read().strip().split('\n')
    except:
        assert file[-3:] == 'tst'
        file = file[:-3] + 'covered.tst'
        lines = open(file, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    examples = [[s for s in l.split('\t')] for l in lines]
    if len(examples[0]) == 2:
        assert 'covered.tst' in file
        for x in examples:
            x.append(x[0]+x[0])

    if do_print:
        print("Number of Examples:", len(examples), '\n')
        print("Examples (0-10):", examples[:10])

    return examples

def getTrainValTestDataInflection(path, lang,
                                  do_test=True, new_schema=False):

    assert lang != 'jap23', 'Special handling for Japanese characters is deleted out of production code; ' \
                            'if need please contact Marc and he can harvest it from older branches.'

    data_train = readUnimorphData(path + lang +'.trn', do_print=False)
    data_val = readUnimorphData(path + lang + '.dev', do_print=False)
    data_test = readUnimorphData(path + lang + '.tst', do_print=False) if do_test else None

    if new_schema: # Has tags and form flipped
        assert '23' in lang, 'Sanity check! Order is lemma tags forms for \'23 but different for other years'
        data_train = [[x[0], x[2], x[1]] for x in data_train]
        data_val = [[x[0], x[2], x[1]] for x in data_val]
        if do_test: data_test = [[x[0], x[2], x[1]] for x in data_test]

    return data_train, data_val, data_test

def tokenize_tags_new(tags):
    tgs = tags.replace('(',';(;').replace(')',';);').replace(',',';')
    splt = tgs.split(';')
    return [x for x in splt if x != '']

@register_model_architecture('transformer', 'inflection_transformer')
def inflection_transformer(args):
    if 'hyper_size' in args:
        def set_hyper_size(edim, ffn, elayers, heads, jlayer, prep_size):
            args.encoder_embed_dim = edim
            args.decoder_embed_dim = args.encoder_embed_dim
            args.encoder_ffn_embed_dim = ffn
            args.decoder_ffn_embed_dim = args.encoder_ffn_embed_dim
            args.encoder_layers = elayers
            args.decoder_layers = args.encoder_layers
            args.encoder_attention_heads = heads
            args.decoder_attention_heads = args.encoder_attention_heads

        if args.hyper_size == 'small':
            set_hyper_size(64, 256, 2,2,1,32)
            assert args.lr == [0.005]
        elif args.hyper_size == 'medium':
            set_hyper_size(128, 512, 3,4,2,128)
            assert args.lr == [0.001]
        elif args.hyper_size == 'large':
            set_hyper_size(256, 1024, 4,8,3,256)
            assert args.lr == [0.001]
        else: assert False, 'Should never get here!'
    else: assert False, 'Currently only support explicitly setting hyper_size'

    args.share_all_embeddings = True # For morph task since input/output vocab is the same.

    args.adam_betas = "(0.9, 0.98)"
    args.clip_norm = 0.0 # 1.0 #
    args.lr_scheduler = 'inverse_sqrt'
    args.warmup_init_lr = 1e-07
    args.dropout = 0.3
    args.weight_decay = 0.0

    args.optimizer = 'adam'

    return bidi_transformer(args)


@register_model_architecture('bidi_transformer', 'bidi_inflection_transformer')
def bidi_inflection_transformer(args):
    return inflection_transformer(args)


