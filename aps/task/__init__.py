from .unsuper import UnsuperEnhTask
from .base import Task
from . import asr, sse

task_cls = {
    "lm": asr.LmXentTask,
    "ctc_xent": asr.CtcXentHybridTask,
    "transducer": asr.TransducerTask,
    "unsuper_enh": UnsuperEnhTask,
    "sisnr": sse.SisnrTask,
    "snr": sse.SnrTask,
    "wa": sse.WaTask,
    "linear_sa": sse.LinearFreqSaTask,
    "mel_sa": sse.MelFreqSaTask,
    "time_linear_sa": sse.LinearTimeSaTask,
    "time_mel_sa": sse.MelTimeSaTask
}
