from .unsupervised_enh import UnsupervisedEnh
from .enh import DCUNet, CRNet, Phasen
from .bss import *
from .toy import ToyRNN

nnet_cls = {
    "unsupervised_enh": UnsupervisedEnh,
    "time_tasnet": TimeConvTasNet,
    "freq_tasnet": FreqConvTasNet,
    "dcunet": DCUNet,
    "crn": CRNet,
    "dccrn": DCCRN,
    "dense_unet": DenseUnet,
    "time_dprnn": TimeDPRNN,
    "freq_dprnn": FreqDPRNN,
    "base_rnn": ToyRNN,
    "phasen": Phasen,
    "freq_xfmr": FreqTorchXfmr
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]