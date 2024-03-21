import torch
import torch.nn as nn
from typing import Optional

try:
    from examples.bidi_decoding.bidi_utils import createGatherers
except:
    from bidi_decoding.bidi_utils import createGatherers

from fairseq.models import (
    register_model,
    register_model_architecture,
    BaseFairseqModel
)
from fairseq.models.transformer import (
    TransformerModel
)

# Use when doing decoder-only task, some methods are needed from time to time
class DummyEncoder(nn.Module):
    def max_positions(self):
        return 1

    def forward_torchscript(self, net_input):
        return None

    def reorder_encoder_out(self, encoder_outs, new_order):
        return None

@register_model("bidi_transformer")
class BidiTransformer(BaseFairseqModel):

    def __init__(self, args, model, r2l_lin_layer, join_lin_layer, order_lin_layer):
        super().__init__()
        self.model = model
        self.r2l_lin_layer = r2l_lin_layer
        self.join_lin_layer = join_lin_layer
        self.order_lin_layer = order_lin_layer

        self.do_order = args.do_order

        self.trg_dictionary = None # will wedge this in later, so can be used in bidi generator (mainly in case of share dec, where we have to replace eos with bos in r2l)
        self.tp = 'fullbidi'

        # DON'T PUT BOOLS HERE BECAUSE CAN CAUSE TROUBLES ON THE DISTRIBUTED.
        # INSTEAD, WEDIGNG THEM INTO JOIN_TRANSFOMER, WHICH DOESNT SEEM TO BE PROBLEMATIC!!!!!!!

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument('--hyper-size',
        #                     help='trick to avoid having to specify individual hyperparameters. sh"b" small, medium, large',
        #                     type=str, required=True)
        pass


    @classmethod
    def build_model(cls, args, task):

        assert args.decoder_embed_dim == args.encoder_embed_dim, 'just check everything if this ends up being false - not sure it all holds.'

        do_order = args.do_order

        model = BidiTransformerModel.build_model(args, task)
        embed_dim = model.decoder.output_embed_dim

        num_vecs = 3 if not do_order else 4
        r2l_lin_layer = nn.Linear(embed_dim, model.decoder.output_projection.out_features)
        nn.init.normal_(
            r2l_lin_layer.weight, mean=0, std=embed_dim ** -0.5
        )
        if num_vecs == 3:
            join_lin_layer = nn.Linear(embed_dim, 2, bias=False)  # bias = False is a remnant of how original fairseq code did it
            nn.init.normal_(join_lin_layer.weight, mean=0, std=embed_dim ** -0.5)
            order_lin_layer=None
        else:
            join_lin_layer = nn.Linear(embed_dim, 2, bias=False)  # bias = False is a remnant of how original fairseq code did it
            nn.init.normal_(join_lin_layer.weight, mean=0, std=embed_dim ** -0.5)
            order_lin_layer = nn.Linear(embed_dim, 2, bias=False)  # bias = False is a remnant of how original fairseq code did it
            nn.init.normal_(order_lin_layer.weight, mean=0, std=embed_dim ** -0.5)

        return cls(args, model, r2l_lin_layer, join_lin_layer, order_lin_layer)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        inputs_l2r,
        inputs_r2l,
        target_dictionary,
        do_rand = False,
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
        self.trg_dictionary = target_dictionary
        assert torch.all(inputs_l2r[:,0] == target_dictionary.eos_index) and torch.all(inputs_r2l[:,0] == target_dictionary.eos_index)
        assert torch.all(inputs_l2r != target_dictionary.bos_index) and torch.all(inputs_r2l != target_dictionary.bos_index) # in share dec case, allos us to replace the first eos with bos

        do_encoder = self.task.src_dict is not None

        if do_encoder:
            enc_out = self.model(src_tokens,
                                     src_lengths,
                                     None, None, return_all_hiddens,
                                                      None,
                                                      None, None, just_encoder=True)
        else: enc_out = None

        lengths = torch.sum(inputs_l2r != target_dictionary.pad_index, dim=1) + 1

        gath0 = [createGatherers(length.item())[0] for length in lengths]
        gath_concat = torch.cat(gath0, dim=0)
        gath_wd_idxes = torch.cat(
            [torch.full((len(gath0[i]),), i, device=inputs_l2r.device) for i in range(len(gath0))])
        max_len = max(lengths)
        if do_rand:
            size = lengths.sum() - lengths.shape[0]
        else:
            size = gath_concat.shape[0]

        join_probs = torch.zeros(size, 2, device=inputs_l2r.device)
        order_probs = torch.zeros(size, 2, device=inputs_l2r.device) if self.do_order else None
        l2r_probs = torch.zeros(size, len(target_dictionary),  device=inputs_l2r.device)
        r2l_probs = torch.zeros(size, len(target_dictionary),  device=inputs_l2r.device)

        join_idxes = torch.zeros(size, device=inputs_l2r.device)
        curr_idx = 0

        for diag in range(max_len - 1):
            result_here, idxes_here = self.doOrigDiagForAdvJorder(enc_out, inputs_l2r, inputs_r2l,
                                                                  gath_concat, gath_wd_idxes, diag, src_lengths, do_rand, False)

            if self.do_order:
                rhere_l2r, rhere_r2l, rhere_join, rhere_order = result_here
                assert rhere_order.shape[0] == idxes_here.shape[0]
            else:
                rhere_l2r, rhere_r2l, rhere_join = result_here
            assert rhere_l2r.shape[0] == idxes_here.shape[0]
            assert rhere_r2l.shape[0] == idxes_here.shape[0]
            assert rhere_join.shape[0] == idxes_here.shape[0]

            l2r_probs[curr_idx:curr_idx + rhere_l2r.shape[0]] = rhere_l2r
            r2l_probs[curr_idx:curr_idx + rhere_r2l.shape[0]] = rhere_r2l
            join_probs[curr_idx:curr_idx + rhere_join.shape[0]] = rhere_join
            if self.do_order: order_probs[curr_idx:curr_idx + rhere_order.shape[0]] = rhere_order

            join_idxes[curr_idx:curr_idx + idxes_here.shape[0]] = idxes_here
            curr_idx += rhere_l2r.shape[0]

        assert size == curr_idx

        if not do_rand:  # shuffle it back so that it's in the original order. that way we know what's the join
            real_join_idxes = torch.zeros(
                join_probs.shape[0]).long()  # have to convert it so f[i] = index in join_probs that what was at i is.
            real_join_idxes[join_idxes.long()] = torch.arange(join_probs.shape[0]).long()
            join_probs = join_probs[real_join_idxes.long(), :]
            if self.do_order: order_probs = order_probs[real_join_idxes.long(), :]
            l2r_probs = l2r_probs[real_join_idxes.long(), :]
            r2l_probs = r2l_probs[real_join_idxes.long(), :]
        else:  # the indexes in join_idxes actually tells us what the join is, so just returning that tuple (think could have been done above too)
            join_probs = (join_probs, join_idxes)





        return l2r_probs, r2l_probs, join_probs, order_probs

    def doOrigDiagForAdvJorder(self, enc_out, inputs_l2r, inputs_r2l, gath_concat, gath_wd_idxes, diag, src_lengths, do_rand=False, return_batch=False):
        diag_bools = gath_concat.sum(dim=1) == diag
        diag_idxes = diag_bools.nonzero().squeeze(1)
        diag_wds = gath_wd_idxes[diag_bools]
        diag_vals = gath_concat[diag_bools]
        if do_rand:
            assert diag_wds.shape[0] % (diag + 1) == 0
            num_words = diag_wds.shape[0] // (diag + 1)
            tmp = torch.randint(0, diag + 1, (num_words,))
            ar = torch.arange(num_words) * (diag+1)
            picks = ar + tmp
            new_arr = torch.full((diag_vals.shape[0], diag_vals.shape[1]), 99999999, device = diag_vals.device)
            new_arr[picks, :] = diag_vals[picks, :]
            diag_vals = new_arr
            num_returns = num_words
        else:
            num_returns = diag_wds.shape[0]

        size = diag+5 if not self.do_order else diag+6 # 2 for diag being off by 2, 2 for each sep, 1 for join, 1 for order

        result_here = torch.zeros(num_returns, size,device=inputs_l2r.device)

        idxes_here = torch.zeros(num_returns, device=inputs_l2r.device)
        diag_wds_concatted = torch.zeros(num_returns, device=inputs_l2r.device)
        curr_idx = 0

        idxes_l2r =  torch.zeros(num_returns, device=inputs_l2r.device)

        for j in range(diag + 1):
            this_cell_bools = diag_vals[:, 0] == j
            diag_wds_here2 = diag_wds[this_cell_bools]
            if this_cell_bools.sum() == 0:
                continue
            up_to_left = diag_vals[this_cell_bools, :][0, 0].item()
            up_to_right = diag_vals[this_cell_bools, :][0, 1].item()

            inputs_here2_l2r = inputs_l2r[diag_wds_here2, :]
            inputs_here2_r2l = inputs_r2l[diag_wds_here2, :]
            chunk_l2r = inputs_here2_l2r[:, :up_to_left + 1]
            chunk_r2l = inputs_here2_r2l[:, :up_to_right + 1]

            sep1 = self.trg_dictionary.indices['<sep>']
            sep2 = self.trg_dictionary.indices['<sep2>']
            clsj = self.trg_dictionary.indices['<cls_j>']
            clso = self.trg_dictionary.indices['<cls_o>']

            if not self.do_order:
                new_chunk = torch.cat([torch.tensor(clsj, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       chunk_l2r,
                                       torch.tensor(sep1, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       torch.tensor(sep2, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       torch.flip(chunk_r2l, dims=[1])], # TO FLIP THE R2L
                                      dim=1)
            else:
                new_chunk = torch.cat([torch.tensor(clsj, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       torch.tensor(clso, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0], 1),
                                       chunk_l2r,
                                       torch.tensor(sep1, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       torch.tensor(sep2, device=chunk_l2r.device).unsqueeze(0).repeat(chunk_l2r.shape[0],1),
                                       torch.flip(chunk_r2l, dims=[1])],# TO FLIP THE R2L
                                      dim=1)

            idxes_here2 = diag_idxes.to(inputs_l2r.device)[this_cell_bools.to(inputs_l2r.device)]

            assert size == new_chunk.shape[1]
            result_here2 = new_chunk

            idxes_here[curr_idx:curr_idx + idxes_here2.shape[0]] = idxes_here2
            diag_wds_concatted[curr_idx:curr_idx + diag_wds_here2.shape[0]] = diag_wds_here2
            assert idxes_here2.shape[0] == result_here2.shape[0]
            result_here[curr_idx:curr_idx+result_here2.shape[0]] = result_here2
            idx_l2r = 1 + (1 if self.do_order else 0) + chunk_l2r.shape[1]
            idxes_l2r[curr_idx:curr_idx+result_here2.shape[0]] = idx_l2r

            curr_idx += idxes_here2.shape[0]

        assert curr_idx == num_returns
        if not return_batch:
            if enc_out is not None:
                eout = {}
                diag_wds_concatted = diag_wds_concatted.long()
                eout['encoder_out'] = [enc_out['encoder_out'][qqq][:,diag_wds_concatted,:] for qqq in range(len(enc_out['encoder_out']))]
                eout['encoder_padding_mask'] = [enc_out['encoder_padding_mask'][qqq][diag_wds_concatted,:] for qqq in range(len(enc_out['encoder_padding_mask']))]
                eout['encoder_embedding'] = [enc_out['encoder_embedding'][qqq][diag_wds_concatted,:,:] for qqq in range(len(enc_out['encoder_embedding']))]
                eout['encoder_states'] = [enc_out['encoder_states'][qqq][:,diag_wds_concatted,:] for qqq in range(len(enc_out['encoder_states']))]
                assert len(enc_out['src_tokens']) == 0
                eout['src_tokens'] = []
                eout['src_lengths'] = [enc_out['src_lengths'][qqq][diag_wds_concatted,:] for qqq in range(len(enc_out['src_lengths']))]
            else: eout = None

            result_here = self.model(None, src_lengths, result_here.long(), eout, False, True, # feats_only=True!!!
            None, None)
            pre_lin_result =result_here[1][0] # 0 b/c dont want the attn weights. ~bsz x L x E
            lin_result_for_l2r=result_here[2] # ~bsz x L x E
        else: assert False, 'havent looked into in a long time'

        lin_result_for_join = self.join_lin_layer(pre_lin_result) # ~bsz x L x 2
        lin_result_for_r2l = self.r2l_lin_layer(pre_lin_result)
        if self.do_order: lin_result_for_order = self.order_lin_layer(pre_lin_result) # ~bsz x L x 2

        # Now have to get the right idxes for the various things
        lin_result_for_join = lin_result_for_join[:,0,:]
        if self.do_order: lin_result_for_order = lin_result_for_order[:,1,:]
        idxes_r2l = 1+idxes_l2r
        lin_result_for_l2r = torch.gather(lin_result_for_l2r, 1,idxes_l2r.unsqueeze(1).repeat(1,lin_result_for_l2r.shape[2]).unsqueeze(1).long()).squeeze(1)
        lin_result_for_r2l = torch.gather(lin_result_for_r2l, 1,idxes_r2l.unsqueeze(1).repeat(1,lin_result_for_r2l.shape[2]).unsqueeze(1).long()).squeeze(1)

        final_result = (lin_result_for_l2r, lin_result_for_r2l, lin_result_for_join) if not self.do_order else (lin_result_for_l2r, lin_result_for_r2l, lin_result_for_join, lin_result_for_order)


        return final_result, idxes_here


class BidiTransformerModel(TransformerModel):

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        encoder_out,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        just_encoder=False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if prev_output_tokens is not None: assert features_only, 'everything i need to do with this requires that'
        if encoder_out is None and src_tokens is not None:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )

        if just_encoder:
            return encoder_out

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            full_context_alignment=True, # SUPER IMPORTANT - don't want causal mask
        )

        lin_out_for_vocab = self.decoder.output_layer(decoder_out[0])

        return encoder_out, decoder_out, lin_out_for_vocab

