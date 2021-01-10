# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

try:
    import kenlm
    kenlm_available = True
except ImportError:
    kenlm_available = False

from aps.conf import load_dict


class NgramLM(object):
    """
    Wrapper for ngram LM, used for beam search

    Args:
        lm: checkpoint of the ngram LM
        vocab_dict: path of the ASR dictionary
    """

    def __init__(self, lm: str, vocab_dict: str) -> None:
        if not kenlm_available:
            raise RuntimeError("import kenlm error, please install kenlm first")
        self.ngram_lm = kenlm.LanguageModel(lm)
        # (int => str)
        vocab = load_dict(vocab_dict, reverse=True)
        self.token = [None] * len(vocab)
        for i, tok in vocab.items():
            if tok == "<eos>":
                tok = "</s>"
            if tok == "<sos>":
                tok = "<s>"
            self.token[i] = tok

    def _step(self, prev_state):
        """
        Args:
            prev_state (State): previous state
        Return:
            score (Tensor): V, LM scores
            state (list[State]), new states
        """
        next_state = [kenlm.State() for _ in range(len(self.token))]
        score = th.tensor([
            self.ngram_lm.BaseScore(prev_state, pred, next_state[i])
            for i, pred in enumerate(self.token)
        ])
        return score, next_state

    def __call__(self, token, state):
        """
        Args:
            token (th.Tensor): V, previous tokens
            state (list[list[State]] or None): LM states
        Return:
            score (Tensor): N x V, LM scores
            state (list[list[State]]), new states
        """
        device = token.device
        token = token.tolist()
        if state is None:
            init_state = kenlm.State()
            self.ngram_lm.BeginSentenceWrite(init_state)
            prev_state = [init_state for _ in range(len(token))]
        else:
            assert len(token) == len(state)
            prev_state = [s[token[i]] for i, s in enumerate(state)]
        scores, states = [], []
        for state in prev_state:
            score, state = self._step(state)
            scores.append(score)
            states.append(state)
        scores = th.stack(scores).to(device)
        return scores, states
