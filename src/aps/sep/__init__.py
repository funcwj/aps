from .unsupervised_enh import UnsupervisedEnh
from .enh import DCUNet, CRNet
from .bss import TimeConvTasNet, FreqConvTasNet, TimeDPRNN, FreqDPRNN, DCCRN
from .toy import ToyRNN

nnet_cls = {
    "unsupervised_enh": UnsupervisedEnh,
    "time_tasnet": TimeConvTasNet,
    "freq_tasnet": FreqConvTasNet,
    "dcunet": DCUNet,
    "crn": CRNet,
    "dccrn": DCCRN,
    "time_dprnn": TimeDPRNN,
    "freq_dprnn": FreqDPRNN,
    "base_rnn": ToyRNN
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]