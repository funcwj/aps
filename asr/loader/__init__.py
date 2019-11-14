from .kaldi_loader import make_dataloader as make_kaldi_loader
from .wav_loader import make_dataloader as make_wave_loader

loader_templ = {"kaldi": make_kaldi_loader, "wav": make_wave_loader}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)