from .am import kaldi, conf, wav
from .lm import bptt, utt
from .ss import chunk

from .am import WaveReader, write_wav

loader_cls = {
    "kaldi": kaldi.DataLoader,
    "conf": conf.DataLoader,
    "wav": wav.DataLoader,
    "enh": chunk.DataLoader,
    "bptt": bptt.DataLoader,
    "utt": utt.DataLoader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_cls:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_cls[fmt](**kwargs)