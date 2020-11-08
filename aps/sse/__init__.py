from .unsupervised_enh import UnsupervisedEnh
from .toy import ToyRNN
from . import enh, bss

sse_nnet_cls = {
    "unsupervised_enh": UnsupervisedEnh,
    "time_tasnet": bss.TimeConvTasNet,
    "freq_tasnet": bss.FreqConvTasNet,
    "dcunet": enh.DCUNet,
    "crn": enh.CRNet,
    "dccrn": bss.DCCRN,
    "dense_unet": bss.DenseUnet,
    "time_dprnn": bss.TimeDPRNN,
    "freq_dprnn": bss.FreqDPRNN,
    "base_rnn": ToyRNN,
    "phasen": enh.Phasen,
    "freq_rel_transformer": bss.FreqRelTransformer
}
