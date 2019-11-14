from .las_asr import LasASR
from .enh_las_asr import EnhLasASR
from .transformer import TransformerASR

nnet_templ = {
    "las": LasASR,
    "enh_las": EnhLasASR,
    "transformer": TransformerASR
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_templ:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_templ[nnet_type]
