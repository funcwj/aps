from . import am, lm, ss

from .am import WaveReader
from .audio import read_wav, write_wav

loader_cls = {
    "am_kaldi": am.kaldi.DataLoader,
    "am_online": am.online.DataLoader,
    "am_wav": am.wav.DataLoader,
    "ss_chunk": ss.chunk.DataLoader,
    "ss_online": ss.online.DataLoader,
    "lm_bptt": lm.bptt.DataLoader,
    "lm_utt": lm.utt.DataLoader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_cls:
        raise RuntimeError(f"Unsupported data-loader type: {fmt}")
    return loader_cls[fmt](**kwargs)