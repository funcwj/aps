from .asr import *
from .sep import *
from .task import Task

task_cls = {
    "lm": LmXentTask,
    "ctc_xent": CtcXentHybridTask,
    "transducer": TransducerTask,
    "unsuper_enh": UnsuperEnhTask,
    "sisnr": SisnrTask,
    "spectra_appro": SaTask
}


def support_task(task, nnet, **kwargs):
    if task not in task_cls:
        raise RuntimeError(f"Unsupported task: {task}")
    return task_cls[task](nnet, **kwargs)