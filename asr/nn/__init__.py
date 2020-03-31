from .las_asr import LasASR
from .enh_las_asr import MvdrLasASR, BeamLasASR
from .transformer_asr import TransformerASR
from .enh_transformer_asr import BeamTransformerASR
from .unsupervised_enh import UnsupervisedEnh
from .transducer_asr import TransformerTransducerASR, TorchTransducerASR
from .lm.rnn import TorchLM

nnet_templ = {
    "torch_lm": TorchLM,
    "las": LasASR,
    "mvdr_las": MvdrLasASR,
    "beam_las": BeamLasASR,
    "transformer": TransformerASR,
    "beam_transformer": BeamTransformerASR,
    "transformer_transducer": TransformerTransducerASR,
    "common_transducer": TorchTransducerASR,
    "unsupervised_enh": UnsupervisedEnh
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_templ:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_templ[nnet_type]
