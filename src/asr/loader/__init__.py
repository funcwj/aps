# wujian@2019

from . import am
from . import lm
from . import enh

loader_templ = {
    "kaldi": am.kaldi.DataLoader,
    "conf": am.conf.DataLoader,
    "wav": am.wav.DataLoader,
    "enh": enh.wav.DataLoader,
    "bptt": lm.bptt.DataLoader,
    "cop": lm.utt.DataLoader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)