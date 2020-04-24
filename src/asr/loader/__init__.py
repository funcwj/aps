from .am import kaldi, conf, wav
from .lm import bptt, utt
from .enh import chunk

loader_templ = {
    "kaldi": kaldi.DataLoader,
    "conf": conf.DataLoader,
    "wav": wav.DataLoader,
    "enh": chunk.DataLoader,
    "bptt": bptt.DataLoader,
    "utt": utt.DataLoader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_templ:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_templ[fmt](**kwargs)