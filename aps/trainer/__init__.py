from .hvd import HvdTrainer
from .ddp import DdpTrainer
from .apex import ApexTrainer

trainer_cls = {"apex": ApexTrainer, "ddp": DdpTrainer, "hvd": HvdTrainer}
