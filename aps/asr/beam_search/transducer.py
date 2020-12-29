#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for transducer based AM
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from queue import PriorityQueue
from typing import Union, List, Dict, Optional


class Node(object):
    """
    Node for usage in best-first beam search
    """

    def __init__(self, score, stats):
        self.score = score
        self.stats = stats

    def __getitem__(self, key):
        return self.stats[key]

    def __lt__(self, other):
        return self.score >= other.score


def _prep_nbest(container: Union[List, PriorityQueue],
                nbest: int,
                len_norm: bool = True,
                blank: int = 0) -> List[Dict]:
    """
    Return nbest hypos from queue or list
    """
    # get nbest
    if isinstance(container, PriorityQueue):
        container = container.queue
    beam_hypos = []
    for node in container:
        trans = [t.item() for t in node["token"]]
        beam_hypos.append({"score": node.score, "trans": trans + [blank]})
    # return best
    nbest_hypos = sorted(beam_hypos,
                         key=lambda n: n["score"] / (len(n["trans"]) - 1
                                                     if len_norm else 1),
                         reverse=True)
    return nbest_hypos[:nbest]


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
                beam: int = 16,
                blank: int = 0,
                nbest: int = 8,
                len_norm: bool = True) -> List[Dict]:
    """
    Beam search (not prefix beam search) algorithm for RNN-T
    Args:
        enc_out: N(=1) x Ti x D
        blank: #vocab_size - 1
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
    if beam > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam}) > vocabulary size")
    if lm and lm.vocab_size < decoder.vocab_size:
        raise RuntimeError("lm.vocab_size < am.vocab_size, "
                           "seems different dictionary is used")
    if blank != decoder.vocab_size - 1:
        raise RuntimeError("Hard code for blank = self.vocab_size - 1 here")

    nbest = min(beam, nbest)

    device = enc_out.device
    blk = th.tensor([blank], dtype=th.int64, device=device)
    # B in Sequence Transduction with Recurrent Neural Networks: Algorithm 1
    list_b = []
    init_node = Node(0.0, {
        "token": [blk],
        "trans": f"{blank}",
        "hidden": None,
        "lm_state": None
    })
    list_b.append(init_node)
    stats = {}
    _, T, _ = enc_out.shape
    for t in range(T):
        # A in Sequence Transduction with Recurrent Neural Networks: Algorithm 1
        list_b = sorted(list_b, key=lambda n: n.score, reverse=True)[:beam]

        queue_a = PriorityQueue()
        for node in list_b:
            queue_a.put_nowait(node)
        list_b = []
        while len(list_b) < beam:
            # pop one (queue_a is updated)
            cur_node = queue_a.get_nowait()
            token = cur_node["token"]
            # make a step
            if cur_node["trans"] in stats:
                dec_out, hidden = stats[cur_node["trans"]]
            else:
                dec_out, hidden = decoder.step(token[-1][..., None],
                                               hidden=cur_node["hidden"])
                stats[cur_node["trans"]] = (dec_out, hidden)

            # prediction: N x 1 x V => V
            pred = decoder.pred(enc_out[:, t], dec_out)[0, 0]
            prob = tf.log_softmax(pred, dim=-1)

            # add terminal node (end with blank)
            score = cur_node.score + prob[blank].item()
            blank_node = Node(
                score, {
                    "token": token,
                    "trans": cur_node["trans"],
                    "hidden": cur_node["hidden"],
                    "lm_state": cur_node["lm_state"]
                })
            list_b.append(blank_node)

            lm_state = None
            if lm and t:
                # 1 x 1 x V (without blank)
                lm_prob, lm_state = lm(token[-1][..., None],
                                       cur_node["lm_state"])
                prob[:-1] += tf.log_softmax(lm_prob[0, -1], dim=-1) * lm_weight

            # extend other nodes
            topk_score, topk_index = th.topk(prob[:-1], beam)
            for i in range(beam):
                score = cur_node.score + topk_score[i].item()
                node = Node(
                    score, {
                        "token":
                            token + [topk_index[None, i]],
                        "trans":
                            cur_node["trans"] + "," + str(topk_index[i].item()),
                        "hidden":
                            hidden,
                        "lm_state":
                            lm_state
                    })
                queue_a.put_nowait(node)

            best_score = queue_a.queue[0].score
            list_b = [n for n in list_b if n.score > best_score]

    return _prep_nbest(list_b, nbest, len_norm=len_norm, blank=blank)
