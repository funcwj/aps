from .las_asr import LasASR
from .transformer import TransformerASR

nnet_templ = {"las": LasASR, "transformer": TransformerASR}


def support_nnet(nnet_type):
    if nnet_type not in nnet_templ:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_templ[nnet_type]
