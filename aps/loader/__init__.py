from . import am, lm, ss

from .audio import read_audio, write_audio, AudioReader

loader_cls = {
    "am_raw": am.raw.DataLoader,
    "am_kaldi": am.kaldi.DataLoader,
    "ss_chunk": ss.chunk.DataLoader,
    "ss_online": ss.online.DataLoader,
    "lm_bptt": lm.bptt.DataLoader,
    "lm_utt": lm.utt.DataLoader
}
