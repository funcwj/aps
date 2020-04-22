# wujian@2019

from .conf import conf_loader
from .wave import wave_loader
from .enhan import enhan_loader
from .kaldi import kaldi_loader
from .token import token_loader
from .corpus import corpus_loader

loader_templ = {
    "kaldi": kaldi_loader,
    "conf": conf_loader,
    "wav": wave_loader,
    "enh": enhan_loader,
    "tok": token_loader,
    "cop": corpus_loader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)