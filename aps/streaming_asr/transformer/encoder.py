#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from aps.asr.base.encoder import EncRetType
from aps.asr.base.attention import padding_mask
from aps.asr.transformer.proj import get_xfmr_proj
from aps.asr.transformer.pose import get_xfmr_pose
from aps.asr.transformer.utils import prep_context_mask
from aps.streaming_asr.transformer.impl import get_xfmr_encoder

from typing import Dict, Optional


class StreamingTransformerEncoder(nn.Module):
    """
    Streaming Transformer encoders.
    """

    def __init__(self,
                 arch: str,
                 input_size: int,
                 output_proj: int = -1,
                 num_layers: int = 6,
                 mhsa_lctx: int = 3,
                 mhsa_rctx: int = 1,
                 proj: str = "conv2d",
                 proj_kwargs: Dict = {},
                 pose: str = "rel",
                 pose_kwargs: Dict = {},
                 arch_kwargs: Dict = {}):
        super(StreamingTransformerEncoder, self).__init__()
        att_dim = arch_kwargs["att_dim"]
        if proj == "none":
            self.proj = None
        else:
            self.proj = get_xfmr_proj(proj, input_size, att_dim, **proj_kwargs)
        if pose != "rel":
            raise ValueError("Now only support rel position encodings")
        if mhsa_rctx + mhsa_lctx == 0:
            raise ValueError(
                f"mhsa_rctx({mhsa_rctx}) + mhsa_lctx({mhsa_lctx}) is zero")
        self.pose = get_xfmr_pose(pose, att_dim // arch_kwargs["nhead"],
                                  **pose_kwargs)
        self.encoder = get_xfmr_encoder(arch, "rel", num_layers, arch_kwargs)
        if output_proj > 0:
            self.outp = nn.Linear(att_dim, output_proj)
        else:
            self.outp = None
        self.lctx, self.rctx = mhsa_lctx, mhsa_rctx

    def reset(self):
        self.encoder.reset()

    def step(self,
             chunk: th.Tensor,
             inj_pose: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            chunk (Tensor): N x T+C x F
            inj_pose (Tensor): T x D
        Return:
            chunk (Tensor): N x 1 x F
        """
        if self.proj is None:
            enc_inp = chunk
        else:
            enc_inp = self.proj(chunk)
        # T x N x D
        enc_inp = enc_inp.transpose(0, 1)
        # 1 x N x D
        enc_out = self.encoder.step(enc_inp, inj_pose=inj_pose)
        # project back
        if self.outp is not None:
            enc_out = self.outp(enc_out)
        # N x 1 x F
        return enc_out.transpose(0, 1)

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Go through projection & transformer encoder layer
        Args:
            inp_pad (Tensor): N x T+C x F
            inp_len (Tensor): N or None
        Return:
            enc_inp (Tensor): N x T x D
            inp_len (Tensor): N or None
        """
        if self.proj is None:
            enc_inp = inp_pad
        else:
            inp_len = self.proj.num_frames(inp_len)
            enc_inp = self.proj(inp_pad)

        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        nframes = enc_inp.shape[1]
        # 2Ti-1 x D
        inj_pose = self.pose(
            th.arange(-nframes + 1, nframes, device=enc_inp.device))
        # src_mask: Ti x Ti
        src_mask = prep_context_mask(nframes,
                                     1,
                                     lctx=self.lctx,
                                     rctx=self.rctx,
                                     device=enc_inp.device)
        # enc_inp: N x Ti x D => Ti x N x D
        enc_inp = enc_inp.transpose(0, 1)
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
                               inj_pose=inj_pose,
                               src_mask=src_mask,
                               src_key_padding_mask=src_pad_mask)
        if self.outp is not None:
            enc_out = self.outp(enc_out)
        # N x Ti x D
        enc_out = enc_out.transpose(0, 1)
        return enc_out, inp_len
