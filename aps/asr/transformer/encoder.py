#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Dict
from aps.asr.base.attention import padding_mask
from aps.asr.base.encoder import EncRetType
from aps.asr.transformer.impl import get_xfmr_encoder
from aps.asr.transformer.pose import get_xfmr_pose
from aps.asr.transformer.proj import get_xfmr_proj
from aps.asr.transformer.utils import prep_context_mask


class TransformerEncoder(nn.Module):
    """
    Transformer based encoders. Currently the arch supports {xfmr|cfmr} and pose supports {abs|rel|xl|conv1d}
    """

    def __init__(self,
                 arch: str,
                 input_size: int,
                 output_proj: int = -1,
                 num_layers: int = 6,
                 lctx: int = -1,
                 rctx: int = -1,
                 chunk_size: int = 1,
                 proj: str = "conv2d",
                 proj_kwargs: Dict = {},
                 pose: str = "abs",
                 pose_kwargs: Dict = {},
                 arch_kwargs: Dict = {}):
        super(TransformerEncoder, self).__init__()
        att_dim = arch_kwargs["att_dim"]
        if proj == "none":
            self.proj = None
        else:
            self.proj = get_xfmr_proj(proj, input_size, att_dim, **proj_kwargs)
        self.pose = get_xfmr_pose(
            pose, att_dim // arch_kwargs["nhead"] if pose == "rel" else att_dim,
            **pose_kwargs)
        self.pose_type = "abs" if pose == "conv1d" else pose
        self.encoder = get_xfmr_encoder(arch, self.pose_type, num_layers,
                                        arch_kwargs)
        self.lctx, self.rctx = lctx, rctx
        self.chunk_size = chunk_size
        if output_proj > 0:
            self.outp = nn.Linear(att_dim, output_proj)
        else:
            self.outp = None

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Go through projection layer
        Args:
            inp_pad: N x Ti x F
            inp_len: N or None
        Return:
            enc_inp: N x Ti x D
            inp_len: N or None
            src_pad_mask: N x Ti or None
        """
        if self.proj is None:
            enc_inp = inp_pad
        else:
            inp_len = self.proj.num_frames(inp_len)
            enc_inp = self.proj(inp_pad)

        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        nframes = enc_inp.shape[1]

        if self.pose_type == "abs":
            # enc_inp: N x Ti x D => Ti x N x D
            enc_inp = self.pose(enc_inp)
            # fake placeholder
            inj_pose = None
        else:
            # enc_inp: N x Ti x D => Ti x N x D
            enc_inp = enc_inp.transpose(0, 1)
            # 2Ti-1 x D
            if self.pose_type == "rel":
                inj_pose = self.pose(
                    th.arange(-nframes + 1, nframes, device=enc_inp.device))
            else:
                inj_pose = self.pose(
                    th.arange(0, 2 * nframes - 1, 1.0, device=enc_inp.device))
        # src_mask: Ti x Ti
        if self.lctx != -1 or self.rctx != -1:
            src_mask = prep_context_mask(nframes,
                                         self.chunk_size,
                                         lctx=self.lctx,
                                         rctx=self.rctx,
                                         device=enc_inp.device)
        else:
            src_mask = None
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
                               inj_pose=inj_pose,
                               src_mask=src_mask,
                               src_key_padding_mask=src_pad_mask)
        if self.outp is not None:
            enc_out = self.outp(enc_out)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len
