#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import List, Union, Tuple

ConvParam = Union[int, List[int]]


class ConvParam(object):
    """
    Maintain parameters of the convolutions
    """

    def __init__(self,
                 kernel_size,
                 stride: int = 1,
                 dilation: int = 1,
                 causal: bool = False,
                 stride_factor: int = 1):
        kernel_context = (kernel_size - 1) // 2
        if causal:
            self.lctx, self.rctx = kernel_context * 2 * dilation, 0
        else:
            self.lctx, self.rctx = kernel_context * dilation, kernel_context * dilation
        self.stride = stride
        self.stride_factor = stride_factor

    def context(self):
        return (self.lctx * self.stride_factor, self.rctx * self.stride_factor)


def compute_conv_context(num_layers: int,
                         kernel: ConvParam,
                         stride: ConvParam,
                         causal: ConvParam = 0,
                         dilation: ConvParam = 1) -> Tuple[int, int, int]:
    """
    Return the left context & right context & stride size of the convolutions on time axis
    """

    def int2list(param, repeat):
        return [param] * repeat if isinstance(param, int) else param

    kernel = int2list(kernel, num_layers)
    stride = int2list(stride, num_layers)
    causal = int2list(causal, num_layers)
    dilation = int2list(dilation, num_layers)

    conv_param = []
    stride_factor = 1
    for i in range(num_layers):
        conv_param.append(
            ConvParam(kernel[i],
                      stride=stride[i],
                      dilation=dilation[i],
                      causal=causal[i],
                      stride_factor=stride_factor))
        stride_factor *= conv_param[-1].stride

    lctx, rctx = 0, 0
    for p in conv_param:
        ctx = p.context()
        lctx += ctx[0]
        rctx += ctx[1]
    return (lctx, rctx, stride_factor)


def test(K, S, D, C):
    lctx, rctx, stride = compute_conv_context(K, S, C, D)
    print(f"lctx: {lctx}")
    print(f"rctx: {rctx}")
    print(f"stride: {stride}")
    T = 100
    channel = 5
    batch = 3
    conv = []
    N = len(K)
    for i in range(N):
        conv.append(
            nn.Conv1d(channel,
                      channel,
                      K[i],
                      stride=S[i],
                      dilation=D[i],
                      padding=0))
    nnet = nn.Sequential(*conv)
    egs_in = th.rand(batch, channel, T)
    egs_out1 = nnet(egs_in)
    egs_out2 = []
    ctx = lctx + rctx + 1
    for t in range(0, T - ctx, stride):
        egs_out2.append(nnet(egs_in[..., t:t + ctx]))
    egs_out2 = th.cat(egs_out2, -1)
    th.testing.assert_allclose(egs_out1, egs_out2)


if __name__ == "__main__":
    K = [3, 5, 7, 3]
    S = [2, 1, 2, 2]
    D = [1, 1, 1, 2]
    C = [False, True, False, False]
    test(K, S, D, C)
