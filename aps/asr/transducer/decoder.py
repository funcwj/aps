#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from queue import PriorityQueue
from typing import List, Dict, Optional, Union, Tuple

from aps.asr.transformer.decoder import prep_sub_mask
from aps.asr.transformer.encoder import ApsTransformerEncoder
from aps.asr.transformer.impl import TransformerTorchEncoderLayer
from aps.asr.transformer.pose import InputSinPosEncoding
from aps.asr.base.attention import padding_mask
from aps.asr.base.layer import OneHotEmbedding, PyTorchRNN


class Node(object):
    """
    Node for usage in best-first beam search
    """

    def __init__(self, score, stats):
        self.score = score
        self.stats = stats

    def __lt__(self, other):
        return self.score >= other.score


def _prep_nbest(container: Union[List, PriorityQueue],
                nbest: int,
                normalized: bool = True,
                blank: int = 0) -> List[Dict]:
    """
    Return nbest hypos from queue or list
    """
    # get nbest
    if isinstance(container, PriorityQueue):
        beam_hypos = []
        while not container.empty():
            node = container.get_nowait()
            trans = [t.item() for t in node.stats["trans"]]
            beam_hypos.append({"score": node.score, "trans": trans + [blank]})
    else:
        beam_hypos = container
    # return best
    nbest_hypos = sorted(beam_hypos,
                         key=lambda n: n["score"] / (len(n["trans"]) - 1
                                                     if normalized else 1),
                         reverse=True)
    return nbest_hypos[:nbest]


class PyTorchRNNDecoder(nn.Module):
    """
    Wrapper for pytorch's RNN Decoder
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512,
                 enc_dim: int = 512,
                 jot_dim: int = 512,
                 dec_rnn: str = "lstm",
                 dec_layers: int = 3,
                 dec_hidden: int = 512,
                 dec_dropout: float = 0.0) -> None:
        super(PyTorchRNNDecoder, self).__init__()
        if embed_size != vocab_size:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        # uni-dir RNNs
        self.decoder = PyTorchRNN(dec_rnn,
                                  embed_size,
                                  dec_hidden,
                                  dec_layers,
                                  dropout=dec_dropout,
                                  bidirectional=False)
        self.enc_proj = nn.Linear(enc_dim, jot_dim, bias=False)
        self.dec_proj = nn.Linear(dec_hidden, jot_dim)
        self.vocab_size = vocab_size
        self.output = nn.Linear(jot_dim, vocab_size, bias=False)

    def forward(self, enc_out: th.Tensor, tgt_pad: th.Tensor) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): N x Ti x D
            tgt_pad (Tensor): N x To+1 (padding blank at time = 0)
        Return:
            output: N x Ti x To+1 x V
        """
        # N x To+1 x E
        tgt_pad = self.vocab_embed(tgt_pad)
        # N x To+1 x D
        dec_out, _ = self.decoder(tgt_pad)
        # N x Ti x To+1 x V
        return self._pred_joint(enc_out, dec_out)

    def _pred_joint(self, enc_out: th.Tensor, dec_out: th.Tensor) -> th.Tensor:
        """
        Joint network prediction
        Args:
            enc_out: N x Ti x D or N x D
            dec_out: N x To+1 x D or N x D
        Return:
            output: N x Ti x To+1 x V or N x 1 x V
        """
        # N x Ti x J or N x J
        enc_out = self.enc_proj(enc_out)
        # N x To+1 x J or N x J
        dec_out = self.dec_proj(dec_out)
        # N x Ti x To+1 x J or N x 1 x J
        add_out = th.tanh(enc_out.unsqueeze(-2) + dec_out.unsqueeze(1))
        # N x Ti x To+1 x V or N x 1 x V
        return self.output(add_out)

    def _step_decoder(self, pred_prev, hidden=None):
        """
        Make one step for decoder
        """
        pred_prev_emb = self.vocab_embed(pred_prev)  # 1 x 1 x E
        dec_out, hidden = self.decoder(pred_prev_emb, hidden)
        return dec_out[:, -1], hidden

    def greedy_search(self, enc_out: th.Tensor, blank: int = 0) -> List[Dict]:
        """
        Greedy search algorithm for RNN-T
        Args:
            enc_out: N x Ti x D
        """
        blk = th.tensor([[blank]], dtype=th.int64, device=enc_out.device)
        dec_out, hidden = self._step_decoder(blk)
        score = 0
        trans = []
        _, T, _ = enc_out.shape
        for t in range(T):
            # 1 x V
            prob = tf.log_softmax(self._pred_joint(enc_out[:, t], dec_out)[0],
                                  dim=-1)
            best_prob, best_pred = th.max(prob, dim=-1)
            score += best_prob.item()
            # not blank
            if best_pred.item() != blank:
                dec_out, hidden = self._step_decoder(best_pred[None, ...],
                                                     hidden=hidden)
                trans += [best_pred.item()]
        return [{"score": score, "trans": [blank] + trans + [blank]}]

    def beam_search(self,
                    enc_out: th.Tensor,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    beam: int = 16,
                    blank: int = 0,
                    nbest: int = 8,
                    normalized: bool = True) -> List[Dict]:
        """
        TODO: re-impl
        Beam search (best first) algorithm for RNN-T
        Args:
            enc_out: N(=1) x Ti x D
        """
        nbest = min(beam, nbest)
        if beam > self.vocab_size:
            raise RuntimeError(f"Beam size({beam}) > vocabulary size")
        if lm and lm.vocab_size < self.vocab_size:
            raise RuntimeError("lm.vocab_size < am.vocab_size, "
                               "seems different dictionary is used")

        dev = enc_out.device
        blk = th.tensor([[blank]], dtype=th.int64, device=dev)
        beam_queue = PriorityQueue()
        init_node = Node(0.0, {
            "trans": [blk],
            "hidden": None,
            "lm_state": None
        })
        beam_queue.put_nowait(init_node)

        _, T, _ = enc_out.shape
        for t in range(T):
            queue_t = beam_queue
            beam_queue = PriorityQueue()
            for _ in range(beam):
                # pop one (queue_t is updated)
                cur_node = queue_t.get_nowait()
                trans = cur_node.stats["trans"]
                # make a step
                # cur_inp = th.tensor([[trans[-1]]], dtype=th.int64, device=dev)
                dec_out, hidden = self._step_decoder(
                    trans[-1], hidden=cur_node.stats["hidden"])
                # predition: 1 x V
                prob = tf.log_softmax(self._pred_joint(enc_out[:, t],
                                                       dec_out)[0],
                                      dim=-1).squeeze()

                # add terminal node (end with blank)
                score = cur_node.score + prob[blank].item()
                blank_node = Node(
                    score, {
                        "trans": trans,
                        "lm_state": cur_node.stats["lm_state"],
                        "hidden": cur_node.stats["hidden"]
                    })
                beam_queue.put_nowait(blank_node)

                lm_state = None
                if lm and t:
                    # 1 x 1 x V (without blank)
                    lm_prob, lm_state = lm(trans[-1],
                                           cur_node.stats["lm_state"])
                    if blank != self.vocab_size - 1:
                        raise RuntimeError(
                            "Hard code for blank = self.vocab_size - 1 here")
                    prob[:-1] += tf.log_softmax(lm_prob[:, -1].squeeze(),
                                                dim=-1) * lm_weight

                # extend other nodes
                topk_score, topk_index = th.topk(prob, beam + 1)
                topk = topk_index.tolist()
                for i in range(beam + 1 if blank in topk else beam):
                    if topk[i] == blank:
                        continue
                    score = cur_node.score + topk_score[i].item()
                    node = Node(
                        score, {
                            "trans": trans + [topk_index[None, i][..., None]],
                            "hidden": hidden,
                            "lm_state": lm_state
                        })
                    queue_t.put_nowait(node)

        return _prep_nbest(beam_queue,
                           nbest,
                           normalized=normalized,
                           blank=blank)

    def beam_search_breadth_first(self,
                                  enc_out,
                                  beam=16,
                                  blank=0,
                                  nbest=8,
                                  normalized=True):
        """
        Beam search (breadth first) algorithm for RNN-T
        Args:
            enc_out: Ti x N(=1) x D
        """
        return None


class TorchTransformerDecoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 vocab_size: int,
                 enc_dim: Optional[int] = None,
                 jot_dim: int = 512,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 num_layers: int = 6,
                 post_norm: bool = True) -> None:
        super(TorchTransformerDecoder, self).__init__()
        self.vocab_embed = nn.Embedding(vocab_size, att_dim)
        self.abs_pos_enc = InputSinPosEncoding(att_dim,
                                               dropout=pos_dropout,
                                               scale_embed=scale_embed)
        decoder_layer = TransformerTorchEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.decoder = ApsTransformerEncoder(decoder_layer,
                                             num_layers,
                                             norm=final_norm)
        self.enc_proj = nn.Linear(enc_dim if enc_dim else att_dim,
                                  jot_dim,
                                  bias=False)
        self.dec_proj = nn.Linear(att_dim, jot_dim)
        self.vocab_size = vocab_size
        self.output = nn.Linear(jot_dim, vocab_size, bias=False)

    def forward(self, enc_out: th.Tensor, tgt_pad: th.Tensor,
                tgt_len: Optional[th.Tensor]) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): Ti x N x D
            tgt_pad (Tensor): N x To+1 (padding blank at time = 1)
            tgt_len (Tensor): N or None
        Return:
            output: N x Ti x To+1 x V
        """
        # N x Ti
        pad_mask = None if tgt_len is None else (padding_mask(tgt_len + 1) == 1)
        # genrarte target masks (-inf/0)
        tgt_mask = prep_sub_mask(tgt_pad.shape[-1], device=tgt_pad.device)
        # To+1 x N x E
        tgt_pad = self.abs_pos_enc(self.vocab_embed(tgt_pad))
        # To+1 x N x D
        dec_out = self.decoder(tgt_pad,
                               src_mask=tgt_mask,
                               src_key_padding_mask=pad_mask)
        return self._pred_joint(enc_out, dec_out)

    def _pred_joint(self, enc_out: th.Tensor, dec_out: th.Tensor) -> th.Tensor:
        """
        Joint network prediction
        Args:
            enc_out: Ti x N x D or 1 x D
            dec_out: To+1 x N x D or 1 x D
        Return:
            output: N x Ti x To+1 x V or N x 1 x V
        """
        enc_out = self.enc_proj(enc_out)
        dec_out = self.dec_proj(dec_out)
        # To+1 x Ti x N x J or 1 x 1 x J
        add_out = th.tanh(enc_out[None, ...] + dec_out[:, None])
        # To+1 x Ti x N x J or 1 x 1 x V
        output = self.output(add_out)
        # N x Ti x To+1 x V or 1 x 1 x V
        if output.dim() == 4:
            output = output.transpose(0, 2)
        if not output.is_contiguous():
            output = output.contiguous()
        return output

    def _step_decoder(self,
                      pred_prev: th.Tensor,
                      prev_embed: Optional[th.Tensor] = None
                     ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Make one step for decoder
        Args:
            pred_prev: 1 x 1
            prev_embed: None or T x 1 x E
        Return:
            dec_out: 1 x D
        """
        t = 0 if prev_embed is None else prev_embed.shape[0]
        # 1 x 1 x E
        pred_prev_emb = self.abs_pos_enc(self.vocab_embed(pred_prev), t=t)
        prev_embed = pred_prev_emb if prev_embed is None else th.cat(
            [prev_embed, pred_prev_emb], dim=0)
        tgt_mask = prep_sub_mask(t + 1, device=pred_prev.device)
        dec_out = self.decoder(prev_embed, mask=tgt_mask)
        return dec_out[-1], prev_embed

    def greedy_search(self, enc_out: th.Tensor, blank: int = 0) -> List[Dict]:
        """
        Greedy search algorithm for RNN-T
        Args:
            enc_out: Ti x N(=1) x D
        """
        blk = th.tensor([[blank]], dtype=th.int64, device=enc_out.device)
        dec_out, prev_embed = self._step_decoder(blk)
        score = 0
        trans = []
        T, _, _ = enc_out.shape
        for t in range(T):
            # 1 x V
            prob = tf.log_softmax(self._pred_joint(enc_out[t], dec_out)[0],
                                  dim=-1)
            best_prob, best_pred = th.max(prob, dim=-1)
            score += best_prob.item()
            # not blank
            if best_pred.item() != blank:
                dec_out, prev_embed = self._step_decoder(best_pred[None, ...],
                                                         prev_embed=prev_embed)
                trans += [best_pred.item()]
        return [{"score": score, "trans": [blank] + trans + [blank]}]

    def beam_search(self,
                    enc_out: th.Tensor,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    beam: int = 16,
                    blank: int = 0,
                    nbest: int = 8,
                    normalized: bool = True) -> List[Dict]:
        """
        TODO: re-impl
        Beam search (best first) algorithm for RNN-T
        Args:
            enc_out: Ti x N(=1) x D
        """
        nbest = min(beam, nbest)
        if beam > self.vocab_size:
            raise RuntimeError(f"Beam size({beam}) > vocabulary size")
        if lm:
            if lm.vocab_size < self.vocab_size:
                raise RuntimeError("lm.vocab_size < am.vocab_size, "
                                   "seems different dictionary is used")

        dev = enc_out.device
        blk = th.tensor([[blank]], dtype=th.int64, device=dev)
        beam_queue = PriorityQueue()
        init_node = Node(0.0, {
            "trans": [blk],
            "prev_embed": None,
            "lm_state": None
        })
        beam_queue.put_nowait(init_node)

        T, _, _ = enc_out.shape
        for t in range(T):
            queue_t = beam_queue
            beam_queue = PriorityQueue()
            for _ in range(beam):
                # pop one
                cur_node = queue_t.get_nowait()
                trans = cur_node.stats["trans"]

                # make a step
                # cur_inp = th.tensor([[trans[-1]]], dtype=th.int64, device=dev)
                dec_out, prev_embed = self._step_decoder(
                    trans[-1], prev_embed=cur_node.stats["prev_embed"])
                # predition: V
                prob = tf.log_softmax(self._pred_joint(enc_out[t], dec_out)[0],
                                      dim=-1).squeeze()

                # add terminal node (end with blank)
                score = cur_node.score + prob[blank].item()
                blank_node = Node(
                    score, {
                        "trans": trans,
                        "lm_state": cur_node.stats["lm_state"],
                        "prev_embed": cur_node.stats["prev_embed"]
                    })
                beam_queue.put_nowait(blank_node)

                lm_state = None
                if lm and t:
                    # 1 x 1 x V (without blank)
                    lm_prob, lm_state = lm(trans[-1],
                                           cur_node.stats["lm_state"])
                    if blank != self.vocab_size - 1:
                        raise RuntimeError(
                            "Hard code for blank = self.vocab_size - 1 here")
                    prob[:-1] += tf.log_softmax(lm_prob[:, -1].squeeze(),
                                                dim=-1) * lm_weight

                # extend other nodes
                topk_score, topk_index = th.topk(prob, beam + 1)
                topk = topk_index.tolist()
                for i in range(beam + 1 if blank in topk else beam):
                    if topk[i] == blank:
                        continue
                    score = cur_node.score + topk_score[i].item()
                    node = Node(
                        score, {
                            "trans": trans + [topk_index[None, i][..., None]],
                            "lm_state": lm_state,
                            "prev_embed": prev_embed
                        })
                    queue_t.put_nowait(node)

        return _prep_nbest(beam_queue,
                           nbest,
                           normalized=normalized,
                           blank=blank)
