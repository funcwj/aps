from .unsupervised_enh import UnsupervisedEnh
from .enh import DCUNet
from .bss import ConvTasNet, DPRNN, Simple

nnet_cls = {
    "unsupervised_enh": UnsupervisedEnh,
    "conv_tasnet": ConvTasNet,
    "dcunet": DCUNet,
    "dprnn": DPRNN,
    "simple": Simple
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]