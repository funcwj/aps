import torch.distributed as dist

try:
    import horovod.torch as hvd
    hvd_available = True
except ImportError:
    hvd_available = False

BACKEND = "none"


def init(backend):
    """
    Set distributed backend
    """
    if backend not in ["torch", "horovod", "none"]:
        raise ValueError(f"Unsupported backend: {backend}")
    BACKEND = backend
    if backend == "horovod":
        hvd.init()


def get_backend():
    """
    Get distributed backend
    """
    return BACKEND


def rank():
    """
    Return rank id
    """
    if BACKEND == "horovod" and not hvd_available:
        raise RuntimeError("horovod not installed!")
    if BACKEND == "none":
        return -1
    elif BACKEND == "torch":
        return dist.get_rank()
    else:
        return hvd.local_rank()


def world_size():
    """
    Return world size
    """
    if BACKEND == "horovod" and not hvd_available:
        raise RuntimeError("horovod not installed!")
    if BACKEND == "none":
        return 0
    elif BACKEND == "torch":
        return dist.get_world_size()
    else:
        return hvd.size()
