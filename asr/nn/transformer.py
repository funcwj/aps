#!/usr/bin/env python

# wujian@2019

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.nn import TransformerDecoder, TransformerDecoderLayer
except:
    raise ImportError("import Transformer module failed")

from .las.attention import padding_mask
from .las.decoder import NEG_INF

IGNORE_ID = -1


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Reference: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos_enc = th.zeros(max_len, embed_dim)
        position = th.arange(0, max_len, dtype=th.float32)
        div_term = th.exp(
            th.arange(0, embed_dim, 2, dtype=th.float32) *
            (-math.log(10000.0) / embed_dim))
        pos_enc[:, 0::2] = th.sin(position[:, None] * div_term)
        pos_enc[:, 1::2] = th.cos(position[:, None] * div_term)
        # Tmax x 1 x D
        self.register_buffer("pos_enc", pos_enc[:, None])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t=0):
        """
        args:
            x: N x T x D 
        return:
            y: T x N x D (keep same as transformer definition)
        """
        _, T, _ = x.shape
        # T x N x D
        x = x.transpose(0, 1)
        # Tmax x 1 x D
        x = x + self.pos_enc[t:t + T, :]
        x = self.dropout(x)
        return x


class LinearEmbedding(nn.Module):
    """
    Linear projection embedding
    """
    def __init__(self, input_size, embed_dim=512):
        super(LinearEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        args:
            x: N x T x F (from asr transform)
        """
        x = self.norm(self.proj(x))
        x = F.relu(x)
        return x


class Conv2dEmbedding(nn.Module):
    """
    2d-conv embedding described in:
        Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition
    """
    def __init__(self, input_size, embed_dim=512):
        super(Conv2dEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, embed_dim, 3, stride=2, padding=1)
        input_size = (input_size - 1) // 2 + 1
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)
        input_size = (input_size - 1) // 2 + 1
        self.proj = nn.Linear(input_size * embed_dim, embed_dim)

    def forward(self, x):
        """
        args:
            x: N x T x F (from asr transform)
        """
        if x.dim() != 3:
            raise RuntimeError(
                f"Conv2dEmbedding expect 3D tensor, got {x.dim()} instead")
        L = x.size(1)
        # N x 1 x T x F => N x A x T' x F'
        x = F.relu(self.conv1(x[:, None]))
        # N x A x T' x F'
        x = F.relu(self.conv2(x))
        # N x T' x A x F'
        x = x.transpose(1, 2)
        N, T, _, _ = x.shape
        x = x.contiguous()
        x = x.view(N, T, -1)
        # N x T' x D
        x = self.proj(x[:, :L // 4])
        return x


class IOEmbedding(nn.Module):
    """
    Kinds of feature embedding layer for ASR tasks
        1) Linear transform
        2) Conv2d transform
        3) Sparse transform
    """
    def __init__(self, embed_type, feature_dim, embed_dim=512, dropout=0.1):
        super(IOEmbedding, self).__init__()
        if embed_type == "linear":
            self.embed = LinearEmbedding(feature_dim, embed_dim=embed_dim)
        elif embed_type == "conv2d":
            self.embed = Conv2dEmbedding(feature_dim, embed_dim=embed_dim)
        elif embed_type == "sparse":
            self.embed = nn.Embedding(feature_dim, embed_dim)
        else:
            raise RuntimeError(f"Unsupported embedding type: {embed_type}")
        self.posencode = PositionalEncoding(embed_dim, dropout=dropout)

    def forward(self, x, t=0):
        """
        args:
            x: N x T x F (from asr transform)
        return:
            y: T' x N x F (to feed transformer)
        """
        y = self.embed(x)
        y = self.posencode(y, t=t)
        return y


class TransformerASR(nn.Module):
    """
    Transformer-based end-to-end ASR
    """
    def __init__(self,
                 input_dim=80,
                 vocab_size=40,
                 sos=-1,
                 eos=-1,
                 ctc=False,
                 asr_transform=None,
                 input_embed="conv2d",
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 num_layers=6):
        super(TransformerASR, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_dim,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout)
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=att_dim,
                                     dropout=0)
        encoder_layer = TransformerEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout)
        decoder_layer = TransformerDecoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        if not eos or not sos:
            raise RuntimeError(f"Unsupported SOS/EOS value: {sos}/{eos}")
        self.sos = sos
        self.eos = eos
        self.asr_transform = asr_transform
        self.input_embed = input_embed
        self.vocab_size = vocab_size
        # if use CTC, eos & sos should be V and V - 1
        self.ctc = nn.Linear(att_dim, vocab_size -
                             2 if sos != eos else vocab_size -
                             1) if ctc else None
        self.output = nn.Linear(att_dim, vocab_size, bias=False)

    def _prep_pad_mask(self, x_len, y_pad):
        """
        Prepare source and target padding masks (-inf/0)
            src_pad_mask: N x Ti
            tgt_pad_mask: N x To+1
        """
        # padding position = True
        src_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # pad sos to y_pad N x To+1
        y_pad = F.pad(y_pad, (1, 0), value=self.sos)
        tgt_mask = y_pad == IGNORE_ID
        y_pad = y_pad.masked_fill(tgt_mask, self.eos)
        return y_pad, src_mask, tgt_mask

    def _prep_sub_mask(self, T, device="cpu"):
        """
        Prepare a square sub-sequence masks (-inf/0)
        egs: for N = 8, output
        tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0., 0., 0.]])
        """
        mask = (th.triu(th.ones(T, T, device=device), diagonal=1) == 1).float()
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x_pad, x_len, y_pad, ssr=0):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
        return:
            outs: N x (To+1) x V
        """
        # feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        if self.input_embed == "conv2d":
            x_len = x_len // 4
        # generate padding masks (True/False)
        y_pad, src_pad_mask, tgt_pad_mask = self._prep_pad_mask(x_len, y_pad)
        # genrarte target masks (-inf/0)
        tgt_mask = self._prep_sub_mask(y_pad.shape[-1], device=y_pad.device)
        # x_emb: N x Ti x D => Ti x N x D
        # src_pad_mask: N x Ti
        x_emb = self.src_embed(x_pad)
        # To+1 x N x E
        y_tgt = self.tgt_embed(y_pad)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               mask=None,
                               src_key_padding_mask=src_pad_mask)
        # CTC
        if self.ctc:
            ctc_branch = self.ctc(enc_out)
            ctc_branch = ctc_branch.transpose(0, 1)
        else:
            ctc_branch = None
        # To+1 x N x D
        dec_out = self.decoder(y_tgt,
                               enc_out,
                               tgt_mask=tgt_mask,
                               memory_mask=None,
                               tgt_key_padding_mask=tgt_pad_mask,
                               memory_key_padding_mask=src_pad_mask)
        dec_out = self.output(dec_out)
        # N x To+1 x D
        dec_out = dec_out.transpose(0, 1).contiguous()
        return dec_out, None, ctc_branch, x_len

    def beam_search(self,
                    x,
                    beam=16,
                    nbest=8,
                    max_len=-1,
                    vectorized=True,
                    normalized=True):
        """
        Beam search for Transformer
        """
        def _trace_back_hypos(point,
                              back_point,
                              hist_token,
                              score,
                              sos=1,
                              eos=2):
            """
            Trace back from current time point
            """
            trans = []
            score = score.item()
            for ptr, tok in zip(back_point[::-1], hist_token[::-1]):
                trans.append(tok[point].item())
                point = ptr[point]
            return {"score": score, "trans": [sos] + trans[::-1] + [eos]}

        with th.no_grad():
            # raw wave
            if self.asr_transform:
                if x.dim() != 1:
                    raise RuntimeError("Now only support for one utterance")
                x, _ = self.asr_transform(x[None, ...], None)
            else:
                if x.dim() != 2:
                    raise RuntimeError("Now only support for one utterance")
                x = x[None, ...]
            # 1 x Ti x D => Ti x 1 x D
            x_emb = self.src_embed(x)
            T, _, _ = x_emb.shape
            # Ti x 1 x D
            enc_out = self.encoder(x_emb)
            # Ti x beam x D
            enc_out = th.repeat_interleave(enc_out, beam, 1)

            if max_len <= 0:
                max_len = T
            else:
                max_len = max(T, max_len)

            nbest = min(beam, nbest)
            if beam > self.vocab_size:
                raise RuntimeError(f"Beam size({beam}) > vocabulary size")

            dev = enc_out.device
            accu_score = th.zeros(beam, device=dev)
            hist_token = []
            back_point = []
            pre_emb = None

            hypos = []
            # step by step
            for t in range(max_len):
                # target mask
                tgt_mask = self._prep_sub_mask(t + 1, device=dev)
                # beam
                if t:
                    point = back_point[-1]
                    cur_emb = self.tgt_embed(hist_token[-1][point][..., None],
                                             t=t)
                    pre_emb = th.cat([pre_emb, cur_emb], dim=0)
                else:
                    point = th.arange(0, beam, dtype=th.int64, device=dev)
                    out = th.tensor([self.sos] * beam,
                                    dtype=th.int64,
                                    device=dev)
                    # 1 x beam x E
                    pre_emb = self.tgt_embed(out[..., None])
                # Tcur - 1 x beam x D
                dec_out = self.decoder(pre_emb, enc_out, tgt_mask=tgt_mask)[-1]
                # beam x V
                dec_out = self.output(dec_out)
                # compute prob: beam x V, nagetive
                prob = F.log_softmax(dec_out, dim=-1)
                # local pruning: beam x beam
                topk_score, topk_token = th.topk(prob, beam, dim=-1)

                if t == 0:
                    # beam
                    accu_score += topk_score[0]
                    token = topk_token[0]
                else:
                    # beam x beam = beam x 1 + beam x beam
                    accu_score = accu_score[..., None] + topk_score
                    # beam*beam => beam
                    accu_score, topk_index = th.topk(accu_score.view(-1),
                                                     beam,
                                                     dim=-1)
                    # point to father's node
                    point = topk_index // beam

                    # beam*beam
                    topk_token = topk_token.view(-1)
                    token = topk_token[topk_index]

                # continue flags
                not_end = (token != self.eos).tolist()

                # process eos nodes
                for i, go_on in enumerate(not_end):
                    if not go_on:
                        hyp = _trace_back_hypos(point[i],
                                                back_point,
                                                hist_token,
                                                accu_score[i],
                                                sos=self.sos,
                                                eos=self.eos)
                        accu_score[i] = NEG_INF
                        hypos.append(hyp)

                # all True
                if sum(not_end) == 0:
                    break

                if len(hypos) >= beam:
                    break

                # add best token
                hist_token.append(token)
                back_point.append(point)

                # process non-eos nodes at the final step
                if t == max_len - 1:
                    for i, go_on in enumerate(not_end):
                        if go_on:
                            hyp = _trace_back_hypos(i,
                                                    back_point,
                                                    hist_token,
                                                    accu_score[i],
                                                    sos=self.sos,
                                                    eos=self.eos)
                            hypos.append(hyp)
            if normalized:
                nbest_hypos = sorted(hypos,
                                     key=lambda n: n["score"] /
                                     (len(n["trans"]) - 1),
                                     reverse=True)
            else:
                nbest_hypos = sorted(hypos,
                                     key=lambda n: n["score"],
                                     reverse=True)
            return nbest_hypos[:nbest]