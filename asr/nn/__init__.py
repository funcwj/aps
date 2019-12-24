from .las_asr import LasASR
from .enh_las_asr import MvdrLasASR, FsLasASR
from .transformer import TransformerASR
from .lm.rnnlm import RNNLM

nnet_templ = {
    "rnnlm": RNNLM,
    "las": LasASR,
    "mvdr_las": MvdrLasASR,
    "fs_las": FsLasASR,
    "transformer": TransformerASR
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_templ:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_templ[nnet_type]
