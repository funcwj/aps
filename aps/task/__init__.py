from .asr import *
from .sse import *
from .unsuper import UnsuperEnhTask
from .base import Task

task_cls = {
    "lm": LmXentTask,
    "ctc_xent": CtcXentHybridTask,
    "transducer": TransducerTask,
    "unsuper_enh": UnsuperEnhTask,
    "sisnr": SisnrTask,
    "snr": SnrTask,
    "wa": WaTask,
    "linear_sa": LinearFreqSaTask,
    "mel_sa": MelFreqSaTask,
    "time_linear_sa": LinearTimeSaTask,
    "time_mel_sa": MelTimeSaTask
}


def support_task(task, nnet, **kwargs):
    if task not in task_cls:
        raise RuntimeError(f"Unsupported task: {task}")
    return task_cls[task](nnet, **kwargs)
