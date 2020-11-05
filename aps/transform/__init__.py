from .asr import FeatureTransform as AsrTransform
from .enh import FeatureTransform as EnhTransform
from .enh import FixedBeamformer, DfTransform

transform_cls = {"asr": AsrTransform, "enh": EnhTransform}
