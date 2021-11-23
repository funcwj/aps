# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Load {am,lm,se} training configurations
"""
import yaml
import codecs

from typing import Dict, List, Tuple
from aps.const import EOS_TOKEN, SOS_TOKEN

required_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf"
]
all_ss_conf_keys = required_keys + ["enh_transform", "cmd_args"]
all_am_conf_keys = required_keys + [
    "enh_transform", "asr_transform", "cmd_args"
]
all_lm_conf_keys = required_keys + ["cmd_args"]
transducer_or_ctc_tasks = ["asr@transducer", "asr@ctc"]


def load_dict(dict_path: str,
              reverse: bool = False,
              required: List[str] = [EOS_TOKEN, SOS_TOKEN]) -> Dict:
    """
    Load the dictionary object
    Args:
        dict_path: path of the vocabulary dictionary
        required: required units in the dict
        reverse: return int:str if true, else str:int
    """
    with codecs.open(dict_path, mode="r", encoding="utf-8") as fd:
        vocab = {}
        for line in fd:
            tok, idx = line.split()
            if reverse:
                vocab[int(idx)] = tok
            else:
                if tok in vocab:
                    raise RuntimeError(
                        f"Duplicated token in {dict_path}: {tok}")
                vocab[tok] = int(idx)
    if not reverse:
        for token in required:
            if token not in vocab:
                raise ValueError(f"Miss token: {token} in {dict_path}")
    return vocab


def dump_dict(dict_path: str, vocab_dict: Dict, reverse: bool = False) -> None:
    """
    Dump the dictionary out
    """
    dict_str = []
    # <int, str> if reverse else <str, int>
    for key, val in vocab_dict.items():
        if reverse:
            val, key = key, val
        dict_str.append(f"{key} {val:d}")
    with codecs.open(dict_path, mode="w", encoding="utf-8") as fd:
        fd.write("\n".join(dict_str))


def check_conf(conf: Dict, required_keys: List[str],
               all_keys: List[str]) -> Dict:
    """
    Check the format of the configurations
    """
    # check the invalid item
    for key in conf.keys():
        if key not in all_keys:
            raise ValueError(f"Get invalid configuration item: {key}")
    # create task_conf if None
    if "task_conf" not in conf:
        conf["task_conf"] = {}
    # check the missing items
    for key in required_keys:
        if key not in conf.keys():
            raise ValueError(f"Miss the item in the configuration: {key}?")
    return conf


def load_ss_conf(yaml_conf: str) -> Dict:
    """
    Load yaml configurations for speech separation/enhancement tasks
    """
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)
    return check_conf(conf, required_keys, all_ss_conf_keys)


def load_lm_conf(yaml_conf: str, dict_path: str) -> Dict:
    """
    Load yaml configurations for language model training
    """
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)
    conf = check_conf(conf, required_keys, all_lm_conf_keys)
    vocab = load_dict(dict_path)
    conf["nnet_conf"]["vocab_size"] = len(vocab)
    return conf, vocab


def load_am_conf(yaml_conf: str, dict_path: str) -> Tuple[Dict, Dict]:
    """
    Load yaml configurations for acoustic model training
    """
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)
    conf = check_conf(conf, required_keys, all_am_conf_keys)
    nnet_conf = conf["nnet_conf"]
    is_transducer_or_ctc = conf["task"] in transducer_or_ctc_tasks

    # load and add dict info
    required_units = [] if is_transducer_or_ctc else [EOS_TOKEN, EOS_TOKEN]
    vocab = load_dict(dict_path, required=required_units)
    nnet_conf["vocab_size"] = len(vocab)
    # Generally we don't use eos/sos in
    if not is_transducer_or_ctc:
        nnet_conf["sos"] = vocab[SOS_TOKEN]
        nnet_conf["eos"] = vocab[EOS_TOKEN]
    # for transducer/CTC
    task_conf = conf["task_conf"]
    ctc_att_hybrid = "ctc_weight" in task_conf and task_conf["ctc_weight"] > 0
    if ctc_att_hybrid or is_transducer_or_ctc:
        task_conf["blank"] = len(vocab)
        # add blank
        nnet_conf["vocab_size"] += 1
        if ctc_att_hybrid:
            nnet_conf["ctc"] = True
    return conf, vocab
