# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import argparse


class StrToBoolAction(argparse.Action):
    """
    Since don't like argparse.store_true or argparse.store_false
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ["true", "y", "yes", "1"]:
            bool_value = True
        elif values in ["false", "n", "no", "0"]:
            bool_value = False
        else:
            raise ValueError(f"Unknown value {values} for --{self.dest}")
        setattr(namespace, self.dest, bool_value)


def get_aps_parser():
    """
    Return default training parser for aps
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to save models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--init",
                        type=str,
                        default="",
                        help="Exist model to initialize model training")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Total batch-size. If multi-process, "
                        "each process gets batch-size/num-process")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
                        help="Number of batches trained per epoch "
                        "(for larger training dataset & "
                        "distributed training)")
    parser.add_argument("--save-interval",
                        type=int,
                        default=-1,
                        help="Interval to save the checkpoint")
    parser.add_argument("--prog-interval",
                        type=int,
                        default=100,
                        help="Interval to report the "
                        "progress of the training")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in dataloader")
    parser.add_argument("--tensorboard",
                        action=StrToBoolAction,
                        default=False,
                        help="Flags to use the tensorboad")
    parser.add_argument("--seed",
                        type=str,
                        default="777",
                        help="Random seed used for random package")
    parser.add_argument("--trainer",
                        type=str,
                        default="ddp",
                        choices=["ddp", "hvd", "apex"],
                        help="Which trainer to use in the backend")
    return parser


class BaseTrainParser(object):
    """
    Parser class for training commands
    """
    parser = get_aps_parser()


class DistributedTrainParser(BaseTrainParser):
    """
    Parser class for distributed training
    """
    parser = get_aps_parser()
    parser.add_argument("--device-ids",
                        type=str,
                        default="0,1",
                        help="Training on which GPU devices")
    parser.add_argument("--distributed",
                        type=str,
                        default="torch",
                        choices=["torch", "horovod"],
                        help="Which distributed backend to use")
    parser.add_argument("--dev-batch-factor",
                        type=int,
                        default=2,
                        help="Use batch_size/dev_batch_factor "
                        "for validation batch size")
