from .att_asr import AttASR
from .enh_att_asr import MvdrAttASR, BeamAttASR
from .transformer_asr import TransformerASR
from .enh_transformer_asr import BeamTransformerASR, MvdrTransformerASR
from .transducer_asr import TransformerTransducerASR, TorchTransducerASR
from .lm.rnn import TorchRNNLM
from .lm.transformer import TorchTransformerLM

nnet_cls = {
    "rnn_lm": TorchRNNLM,
    "transformer_lm": TorchTransformerLM,
    "las": AttASR,
    "mvdr_att": MvdrAttASR,
    "beam_att": BeamAttASR,
    "transformer": TransformerASR,
    "beam_transformer": BeamTransformerASR,
    "mvdr_transformer": MvdrTransformerASR,
    "transformer_transducer": TransformerTransducerASR,
    "common_transducer": TorchTransducerASR
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]
