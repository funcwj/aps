from os import environ
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
    if backend == "torch":
        for env in ["LOCAL_RANK", "WORLD_SIZE"]:
            if "LOCAL_RANK" not in environ:
                raise RuntimeError(
                    f"Not found in {env} environments, using python "
                    "-m torch.distributed.launch to launch the command")
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                rank=int(environ["LOCAL_RANK"]),
                                world_size=int(environ["WORLD_SIZE"]))


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
