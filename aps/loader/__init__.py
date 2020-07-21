from . import am, lm, ss

from .audio import read_wav, write_wav, WaveReader

loader_cls = {
    "am_wav": am.wav.DataLoader,
    "am_kaldi": am.kaldi.DataLoader,
    "ss_chunk": ss.chunk.DataLoader,
    "ss_online": ss.online.DataLoader,
    "lm_bptt": lm.bptt.DataLoader,
    "lm_utt": lm.utt.DataLoader
}


def support_loader(fmt="wav", **kwargs):
    if fmt not in loader_cls:
        raise RuntimeError(f"Unsupported DataLoader type: {fmt}")
    return loader_cls[fmt](**kwargs)