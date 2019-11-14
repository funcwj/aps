from .asr import FeatureTransform as AsrTransform
from .enh import FeatureTransform as EnhTransform

trans_templ = {"asr": AsrTransform, "enh": EnhTransform}


def support_transform(trans_type):
    if trans_type not in trans_templ:
        raise RuntimeError(f"Unsupported transform type: {trans_type}")
    return trans_templ[trans_type]
