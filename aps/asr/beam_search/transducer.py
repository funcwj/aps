#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for transducer based AM
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import List, Dict, Optional
from aps.utils import get_logger

logger = get_logger(__name__)


class Node(object):
    """
    Beam node for RNNT beam search
    """

    def __init__(self, score: th.Tensor, stats: Dict) -> None:
        self.score = score
        self.stats = stats

    def __getitem__(self, key):
        return self.stats[key]


def merge_hypos(hypos_list: List[Node]) -> List[Node]:
    """
    Merge the hypos that has the same prefix
    """
    merge_dict = {}
    for hypos in hypos_list:
        prefix_str = hypos["hashid"]
        if prefix_str in merge_dict:
            merge_dict[prefix_str].score = th.logaddexp(
                hypos.score, merge_dict[prefix_str].score)
        else:
            merge_dict[prefix_str] = hypos
    merge_list = [value for _, value in merge_dict.items()]
    return sorted(merge_list, key=lambda n: n.score, reverse=True)


def greedy_search(decoder: nn.Module,
                  enc_out: th.Tensor,
                  blank: int = 0) -> List[Dict]:
    """
    Greedy search algorithm for RNN-T
    Args:
        enc_out: N x Ti x D
    """
    if blank < 0:
        raise RuntimeError(f"Invalid blank ID: {blank:d}")
    N, T, _ = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if not hasattr(decoder, "pred"):
        raise RuntimeError("Function pred should defined in decoder network")

    blk = th.tensor([[blank]], dtype=th.int64, device=enc_out.device)
    dec_out, hidden = decoder.step(blk)
    score = 0
    trans = []
    for t in range(T):
        # 1 x V
        prob = tf.log_softmax(decoder.pred(enc_out[:, t], dec_out)[0], dim=-1)
        best_prob, best_pred = th.max(prob, dim=-1)
        score += best_prob.item()
        # not blank
        if best_pred.item() != blank:
            dec_out, hidden = decoder.step(best_pred[None, ...], hidden=hidden)
            trans += [best_pred.item()]
    return [{"score": score, "trans": [blank] + trans + [blank]}]


def beam_search(decoder: nn.Module,
                enc_out: th.Tensor,
                lm: Optional[nn.Module] = None,
                lm_weight: float = 0,
                beam_size: int = 16,
                blank: int = 0,
                nbest: int = 8,
                len_norm: bool = True) -> List[Dict]:
    """
    Beam search (not prefix beam search) algorithm for RNN-T
    Args:
        enc_out: N(=1) x Ti x D
        blank: #vocab_size - 1
    """
    N, T, _ = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if not hasattr(decoder, "pred"):
        raise RuntimeError("Function pred should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")
    if lm and lm.vocab_size < decoder.vocab_size:
        raise RuntimeError("lm.vocab_size < am.vocab_size, "
                           "seems different dictionary is used")
    if blank != decoder.vocab_size - 1:
        raise RuntimeError("Hard code for blank = self.vocab_size - 1 here")

    nbest = min(beam_size, nbest)

    device = enc_out.device
    blk = th.tensor([blank], dtype=th.int64, device=device)
    init_node = Node(th.tensor(0.0), {
        "hashid": f"{blk}",
        "prefix": [blk],
        "hidden": None,
    })
    # list_a, list_b: A, B in Sequence Transduction with Recurrent Neural Networks: Algorithm 1
    list_b = [init_node]
    for t in range(T):
        # merge hypos, return in order
        list_a = merge_hypos(list_b)
        # logger.info(f"--- merge hypos: {len(list_b)} -> {len(list_a)}")
        list_b = []
        # for _ in range(beam_size):
        while True:
            # pop the best node
            cur_node = list_a[0]
            list_a = list_a[1:]

            prefix_str = cur_node["hashid"]
            prefix_tok = cur_node["prefix"]

            # decoder step
            dec_out, hidden = decoder.step(prefix_tok[-1][..., None],
                                           hidden=cur_node["hidden"])

            # predict: N x 1 x V => V
            pred = decoder.pred(enc_out[:, t], dec_out)[0, 0]
            prob = tf.log_softmax(pred, dim=-1)

            # add terminal node (end with blank)
            score = cur_node.score + prob[blank]
            blank_node = Node(
                score, {
                    "hashid": prefix_str,
                    "prefix": prefix_tok,
                    "hidden": cur_node["hidden"],
                })
            list_b.append(blank_node)

            # extend other nodes
            topk_score, topk_index = th.topk(prob[:-1], beam_size)
            for i in range(beam_size):
                score = cur_node.score + topk_score[i]
                node = Node(
                    score, {
                        "hashid": prefix_str + f",{topk_index[i].item()}",
                        "prefix": prefix_tok + [topk_index[None, i]],
                        "hidden": hidden
                    })
                list_a.append(node)
            # sort A
            list_a = sorted(list_a, key=lambda n: n.score, reverse=True)
            # while B contains less than W elements more probable than the most probable in A
            cur_list_b = [n for n in list_b if n.score > list_a[0].score]
            if len(cur_list_b) >= beam_size:
                list_b = cur_list_b
                break

    final_hypos = [{
        "score": n.score.item() / (len(n["prefix"]) if len_norm else 1),
        "trans": [t.item() for t in n["prefix"]] + [blank]
    } for n in merge_hypos(list_b)]
    # return best
    nbest_hypos = sorted(final_hypos, key=lambda n: n["score"], reverse=True)
    return nbest_hypos[:nbest]
