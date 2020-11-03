from .att import AttASR
from .enh_att import MvdrAttASR, BeamAttASR
from .transformers import TransformerASR
from .enh_transformers import BeamTransformerASR, MvdrTransformerASR
from .transducers import TransformerTransducerASR, TorchTransducerASR
from .lm.rnn import TorchRNNLM
from .lm.transformer import TorchTransformerLM

asr_nnet_cls = {
    "rnn_lm": TorchRNNLM,
    "transformer_lm": TorchTransformerLM,
    "att": AttASR,
    "mvdr_att": MvdrAttASR,
    "beam_att": BeamAttASR,
    "transformer": TransformerASR,
    "beam_transformer": BeamTransformerASR,
    "mvdr_transformer": MvdrTransformerASR,
    "transformer_transducer": TransformerTransducerASR,
    "common_transducer": TorchTransducerASR
}
