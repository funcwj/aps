from .unsupervised_enh import UnsupervisedEnh
from .enh import DCUNet, CRNet, Phasen
from .bss import *
from .toy import ToyRNN

sse_nnet_cls = {
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
    "freq_rel_transformer": FreqRelTransformer
}
