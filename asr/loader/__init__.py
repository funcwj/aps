from .kaldi_loader import make_kaldi_loader
from .wav_loader import make_wave_loader
from .token_loader import make_token_loader
from .conf_loader import make_online_loader

loader_templ = {
    "kaldi": make_kaldi_loader,
    "online": make_online_loader,
    "wav": make_wave_loader,
    "token": make_token_loader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)