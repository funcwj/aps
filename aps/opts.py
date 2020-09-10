import argparse


class StrToBoolAction(argparse.Action):
    """
    Since argparse.store_true is not very convenient
    """

    def __call__(self, parser, namespace, values, option_string=None):

        def str2bool(value):
            if value.lower() in ["true", "y", "yes", "1"]:
                return True
            elif value in ["false", "n", "no", "0"]:
                return False
            else:
                raise ValueError

        try:
            setattr(namespace, self.dest, str2bool(values))
        except ValueError:
            raise Exception(f"Unknown value {values} for --{self.dest}")


class BaseTrainParser(object):
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
                        default=4000,
                        help="Number of batches trained per epoch "
                        "(for larger training dataset & distributed training)")
    parser.add_argument("--save-interval",
                        type=int,
                        default=-1,
                        help="Interval to save the checkpoint")
    parser.add_argument("--prog-interval",
                        type=int,
                        default=100,
                        help="Interval to report the progress of the training")
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
