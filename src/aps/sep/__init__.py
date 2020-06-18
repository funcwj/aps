from .unsupervised_enh import UnsupervisedEnh
from .enh import DCUNet, CRNet
from .bss import ConvTasNet, DPRNN
from .toy import TimeDomainToyRNN, FreqDomainToyRNN

nnet_cls = {
    "unsupervised_enh": UnsupervisedEnh,
    "conv_tasnet": ConvTasNet,
    "dcunet": DCUNet,
    "crn": CRNet,
    "dprnn": DPRNN,
    "time_toy": TimeDomainToyRNN,
    "freq_toy": FreqDomainToyRNN
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]