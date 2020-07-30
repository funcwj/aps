from .asr import FeatureTransform as AsrTransform
from .enh import FeatureTransform as EnhTransform
from .enh import FixedBeamformer, DfTransform

trans_cls = {"asr": AsrTransform, "enh": EnhTransform}


def support_transform(trans_type):
    if trans_type not in trans_cls:
        raise RuntimeError(f"Unsupported transform type: {trans_type}")
    return trans_cls[trans_type]
