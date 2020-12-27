#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Give some custom RNNs implementation
Reference:
    https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
"""
import warnings
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import torch.nn.init as init
import torch.jit as jit

from typing import Optional, Tuple, List

LstmHiddenType = Tuple[th.Tensor, th.Tensor]


@th.jit.script
def stack_hidden(hiddens: List[LstmHiddenType]) -> LstmHiddenType:
    """
    Stack list of LSTM hidden states
    """
    h = th.stack([hidden[0] for hidden in hiddens])
    c = th.stack([hidden[1] for hidden in hiddens])
    return (h, c)


class LSTMProjCell(jit.ScriptModule):
    """
    LSTM cell with a projection layer
    """

    def __init__(self, input_size: int, hidden_size: int,
                 project_size: int) -> None:
        super(LSTMProjCell, self).__init__()
        self.weight_ih = nn.Parameter(th.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(th.randn(4 * hidden_size, project_size))
        self.weight_hr = nn.Parameter(th.randn(project_size, hidden_size))
        self.bias_ih = nn.Parameter(th.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(th.randn(4 * hidden_size))
        self.repr = f"{input_size}, {hidden_size}, project={project_size}"
        self.reset_parameters()

    def extra_repr(self) -> str:
        return self.repr

    def reset_parameters(self) -> None:
        k = (1.0 / self.weight_hr.shape[-1])**0.5
        init.uniform_(self.weight_ih, -k, k)
        init.uniform_(self.weight_hh, -k, k)
        init.uniform_(self.weight_hr, -k, k)
        init.uniform_(self.bias_ih, -k, k)
        init.uniform_(self.bias_ih, -k, k)

    def init_hidden(self, batch_size: int):
        """
        Return zero hidden state
        """
        hy = th.zeros(batch_size,
                      self.weight_hr.shape[0],
                      device=self.weight_hr.device)
        cy = th.zeros(batch_size,
                      self.weight_hr.shape[-1],
                      device=self.weight_hr.device)
        return (hy, cy)

    @jit.script_method
    def forward(self, inp: th.Tensor,
                state: LstmHiddenType) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x D (input size)
            state ([Tensor, Tensor]): (N x P, N x H)
        Return:
            out (Tensor): N x P
            state ([Tensor, Tensor]): (N x P, N x H)
        """
        hx, cx = state
        gates = th.mm(inp, self.weight_ih.t()) + self.bias_ih + th.mm(
            hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = th.sigmoid(ingate)
        forgetgate = th.sigmoid(forgetgate)
        cellgate = th.tanh(cellgate)
        outgate = th.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * th.tanh(cy)
        hy = th.mm(hy, self.weight_hr.t())
        return hy, (hy, cy)


class LSTMLnCell(jit.ScriptModule):
    """
    LSTM cell with layer normalization
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(LSTMLnCell, self).__init__()
        # parameters for LSTM
        self.weight_ih = nn.Parameter(th.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(th.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(th.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(th.randn(4 * hidden_size))
        # layers for layernorm
        self.ln_i = nn.LayerNorm(4 * hidden_size)
        self.ln_h = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.repr = f"{input_size}, {hidden_size}, layer_norm=True"
        self.reset_parameters()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.repr})"

    def reset_parameters(self) -> None:
        k = (1.0 / self.weight_hh.shape[-1])**0.5
        init.uniform_(self.weight_ih, -k, k)
        init.uniform_(self.weight_hh, -k, k)
        init.uniform_(self.bias_ih, -k, k)
        init.uniform_(self.bias_ih, -k, k)

    def init_hidden(self, batch_size: int):
        """
        Return zero hidden state
        """
        hy = th.zeros(batch_size,
                      self.weight_hh.shape[-1],
                      device=self.weight_hh.device)
        cy = th.zeros(batch_size,
                      self.weight_hh.shape[-1],
                      device=self.weight_hh.device)
        return (hy, cy)

    @jit.script_method
    def forward(self, inp: th.Tensor,
                state: LstmHiddenType) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x D (input size)
            state ([Tensor, Tensor]): (N x H, N x H)
        Return:
            out (Tensor): N x H
            state ([Tensor, Tensor]): (N x H, N x H)
        """
        hx, cx = state
        ih = self.ln_i(th.mm(inp, self.weight_ih.t()) + self.bias_ih)
        hh = self.ln_h(th.mm(hx, self.weight_hh.t()) + self.bias_hh)
        gates = ih + hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = th.sigmoid(ingate)
        forgetgate = th.sigmoid(forgetgate)
        cellgate = th.tanh(cellgate)
        outgate = th.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        cy = self.ln_c(cy)
        hy = outgate * th.tanh(cy)
        return hy, (hy, cy)


class LSTMLnProjCell(jit.ScriptModule):
    """
    LSTM cell with layer normalization & projection layer
    """

    def __init__(self, input_size: int, hidden_size: int,
                 project_size: int) -> None:
        super(LSTMLnProjCell, self).__init__()
        # parameters for LSTM
        self.weight_ih = nn.Parameter(th.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(th.randn(4 * hidden_size, project_size))
        self.weight_hr = nn.Parameter(th.randn(project_size, hidden_size))
        self.bias_ih = nn.Parameter(th.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(th.randn(4 * hidden_size))
        # layers for layernorm
        self.ln_i = nn.LayerNorm(4 * hidden_size)
        self.ln_h = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.repr = (f"{input_size}, {hidden_size}, " +
                     f"project={project_size}, layer_norm=True")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.repr})"

    def reset_parameters(self) -> None:
        k = 1.0 / (self.weight_hr.shape[-1])**0.5
        init.uniform_(self.weight_ih, -k, k)
        init.uniform_(self.weight_hh, -k, k)
        init.uniform_(self.weight_hr, -k, k)
        init.uniform_(self.bias_ih, -k, k)
        init.uniform_(self.bias_ih, -k, k)

    def init_hidden(self, batch_size: int):
        """
        Return zero hidden state
        """
        hy = th.zeros(batch_size,
                      self.weight_hr.shape[0],
                      device=self.weight_hr.device)
        cy = th.zeros(batch_size,
                      self.weight_hr.shape[-1],
                      device=self.weight_hr.device)
        return (hy, cy)

    @jit.script_method
    def forward(self, inp: th.Tensor,
                state: LstmHiddenType) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x D (input size)
            state ([Tensor, Tensor]): (N x P, N x H)
        Return:
            out (Tensor): N x P
            state ([Tensor, Tensor]): (N x P, N x H)
        """
        hx, cx = state
        ih = self.ln_i(th.mm(inp, self.weight_ih.t()) + self.bias_ih)
        hh = self.ln_h(th.mm(hx, self.weight_hh.t()) + self.bias_hh)
        gates = ih + hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = th.sigmoid(ingate)
        forgetgate = th.sigmoid(forgetgate)
        cellgate = th.tanh(cellgate)
        outgate = th.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        cy = self.ln_c(cy)
        hy = outgate * th.tanh(cy)
        hy = th.mm(hy, self.weight_hr.t())
        return hy, (hy, cy)


class UniLSTMLayer(jit.ScriptModule):
    """
    Uni-directional LSTM layer
    """

    def __init__(self, cell: jit.ScriptModule, dropout: float = 0.1) -> None:
        super(UniLSTMLayer, self).__init__()
        self.cell = cell
        self.dropout = dropout

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cell.repr}, dropout={self.dropout:.2f})"

    def init_hidden(self, batch_size: int):
        """
        Return zero hidden state
        """
        return self.cell.init_hidden(batch_size)

    @jit.script_method
    def forward(self,
                inp: th.Tensor,
                state: LstmHiddenType,
                reverse: bool = True) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x T x D
            state ([Tensor, Tensor]): [N x ..., N x H]
        Return:
            out (Tensor): N x T x (P|H)
            state ([Tensor, Tensor]): [N x ..., N x H]
        """
        # N x T x F
        inputs = inp.unbind(1)
        if reverse:
            inputs[::-1]
        outputs = th.jit.annotate(List[th.Tensor], [])
        for inp_t in inputs:
            out, state = self.cell(inp_t, state)
            outputs += [out]
        outputs = th.stack(outputs, dim=1)
        return outputs, state


class BidLSTMLayer(jit.ScriptModule):
    """
    Bi-directional LSTM layer
    """

    def __init__(self, cells: List[jit.ScriptModule]) -> None:
        super(BidLSTMLayer, self).__init__()
        self.cell_forward = cells[0]
        self.cell_reverse = cells[1]

    def init_hidden(self, batch_size: int):
        """
        Return zero hidden state
        """
        forward = self.cell_forward.init_hidden(batch_size)
        reverse = self.cell_reverse.init_hidden(batch_size)
        return stack_hidden([forward, reverse])

    @jit.script_method
    def forward(self, inp: th.Tensor,
                state: LstmHiddenType) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x T x D
            state ([Tensor, Tensor]): [2 x N x ..., 2 x N x H]
        Return:
            out (Tensor): N x T x (2P|2H)
            state ([Tensor, Tensor]): [2 x N x ..., 2 x N x H]
        """
        h, c = state
        forward_out, forward_hx = self.cell_forward(inp, (h[0], c[0]),
                                                    reverse=True)
        reverse_out, reverse_hx = self.cell_reverse(inp, (h[1], c[1]),
                                                    reverse=False)
        output = th.cat([forward_out, reverse_out], -1)
        state = stack_hidden([forward_hx, reverse_hx])
        return output, state


def create_lstm_layer(input_size: int,
                      hidden_size: int,
                      dropout: float = 0.0,
                      project: Optional[int] = None,
                      layer_norm: bool = False,
                      bidirectional: bool = True):
    """
    Return the custom lstm layer
    """
    if not layer_norm and project is None:
        raise RuntimeError("In this case, please use PyTorch's LSTM layer")
    number = 2 if bidirectional else 1
    if project:
        inst = LSTMLnProjCell if layer_norm else LSTMProjCell
        cell = [inst(input_size, hidden_size, project) for _ in range(number)]
    else:
        cell = [LSTMLnCell(input_size, hidden_size) for _ in range(number)]
    if bidirectional:
        return BidLSTMLayer([UniLSTMLayer(c, dropout=dropout) for c in cell])
    else:
        return UniLSTMLayer(cell[0], dropout=dropout)


class LSTM(jit.ScriptModule):
    """
    LSTM that supports LSTMP, layer normalization variants
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0,
                 project: Optional[int] = None,
                 num_layers: int = 1,
                 layer_norm: bool = False,
                 bidirectional: bool = True) -> None:
        super(LSTM, self).__init__()
        if num_layers == 1 and dropout != 0:
            warnings.warn("Got one layer LSTM and we don't apply the dropout")
        self.bidirectional = bidirectional
        layers = []
        for i in range(num_layers):
            if i != 0:
                input_size = hidden_size if project is None else project
                if bidirectional:
                    input_size *= 2
            layers.append(
                create_lstm_layer(input_size,
                                  hidden_size,
                                  project=project,
                                  dropout=0 if i == num_layers - 1 else dropout,
                                  layer_norm=layer_norm,
                                  bidirectional=bidirectional))
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        self.dropout = dropout

    def init_hidden(self, batch_size: int) -> LstmHiddenType:
        """
        Return hidden state: L x N x H or L x 2 x N x H
        """
        # [(h, x), ..., (h, x)]
        hiddens = [cell.init_hidden(batch_size) for cell in self.layers]
        # L x N x H or L x 2 x N x H
        return stack_hidden(hiddens)

    @jit.script_method
    def forward(
        self,
        inp: th.Tensor,
        hx: Optional[LstmHiddenType] = None
    ) -> Tuple[th.Tensor, LstmHiddenType]:
        """
        Args:
            inp (Tensor): N x T x D
            hx ([Tensor, Tensor]): [L*(2|1) x N x ..., L*(2|1) x N x ...]
        Return:
            inp (Tensor): N x T x (P|H|2P|2H)
            hx ([Tensor, Tensor]): [L*(2|1) x N x ..., L*(2|1) x N x ...]
        """
        N, _, _ = inp.shape
        if hx is None:
            hx = self.init_hidden(inp.shape[0])
        else:
            if self.bidirectional:
                h, c = hx
                h = h.view(self.num_layers, 2, N, -1)
                c = c.view(self.num_layers, 2, N, -1)
                hx = (h, c)

        states = th.jit.annotate(List[LstmHiddenType], [])
        for index, layer in enumerate(self.layers):
            inp, state = layer(inp, (hx[0][index], hx[1][index]))
            if index != self.num_layers - 1:
                inp = tf.dropout(inp,
                                 p=self.dropout,
                                 training=self.training,
                                 inplace=False)
            states.append(state)
        # L x N x ... or L x 2 x N x ...
        h, c = stack_hidden(states)
        if self.bidirectional:
            h = h.view(self.num_layers * 2, N, -1)
            c = c.view(self.num_layers * 2, N, -1)
        return inp, (h, c)


def test_lstm_proj():
    for num_layers in [1, 2]:
        for bidirectional in [True, False]:
            lstm = LSTM(80,
                        512,
                        project=256,
                        dropout=0.2,
                        bidirectional=bidirectional,
                        num_layers=num_layers,
                        layer_norm=False)
            print(lstm)
            x = th.rand(10, 100, 80)
            y, _ = lstm(x)
            print(y.shape)


def test_lstm_ln():
    for num_layers in [1, 2]:
        for bidirectional in [True, False]:
            lstm = LSTM(80,
                        512,
                        project=None,
                        dropout=0.2,
                        bidirectional=bidirectional,
                        num_layers=num_layers,
                        layer_norm=True)
            print(lstm)
            x = th.rand(10, 100, 80)
            y, _ = lstm(x)
            print(y.shape)


def test_lstm_ln_proj():
    for num_layers in [1, 2]:
        for bidirectional in [True, False]:
            lstm = LSTM(80,
                        512,
                        project=256,
                        dropout=0.2,
                        bidirectional=bidirectional,
                        num_layers=num_layers,
                        layer_norm=True)
            print(lstm)
            x = th.rand(10, 100, 80)
            y, _ = lstm(x)
            print(y.shape)


if __name__ == "__main__":
    test_lstm_proj()
    test_lstm_ln()
    test_lstm_ln_proj()
