# wujian@2019

from .kaldi_loader import make_kaldi_loader
from .wav_loader import make_wav_loader
from .enh_loader import make_enh_loader
from .tok_loader import make_tok_loader
from .conf_loader import make_online_loader

loader_templ = {
    "kaldi": make_kaldi_loader,
    "online": make_online_loader,
    "wav": make_wav_loader,
    "enh": make_enh_loader,
    "tok": make_tok_loader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)