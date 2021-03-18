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


def get_aps_train_parser():
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
                        help="Number of the training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="The directory to save the checkpoints")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Resume training from the existed checkpoint")
    parser.add_argument("--init",
                        type=str,
                        default="",
                        help="Initialize the model using existed checkpoint")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of the batch-size. If distributed "
                        "training is used, "
                        "each process gets #batch_size/#num_process")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
                        help="Number of the batches trained per epoch "
                        "(for larger training dataset & distributed training)")
    parser.add_argument("--save-interval",
                        type=int,
                        default=-1,
                        help="We save the checkpoint per #save_interval epochs")
    parser.add_argument("--prog-interval",
                        type=int,
                        default=100,
                        help="Report the progress of the training "
                        "per #prog_interval batches")
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


def get_asr_base_parser():
    """
    Return base parser for ASR related commands
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index for wav.scp")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the input audio")
    parser.add_argument("--am",
                        type=str,
                        required=True,
                        help="Checkpoint directory of the AM")
    parser.add_argument("--am-tag",
                        type=str,
                        default="best",
                        help="Tag name to load the checkpoint: (tag).pt.tar")
    parser.add_argument("--space",
                        type=str,
                        default="",
                        help="Space symbol to inserted between the "
                        "model units (if needed)")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--dict",
                        type=str,
                        default="",
                        help="Dictionary file (if needed)")
    parser.add_argument("--spm",
                        type=str,
                        default="",
                        help="Path of the sentencepiece model "
                        "(if we use subword unit)")
    return parser


def get_aps_decode_parser():
    """
    Return default decoding parser for aps
    """
    parser = get_asr_base_parser()
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Source scripts of the feature or audio")
    parser.add_argument("best",
                        type=str,
                        help="Output file of the decoded results (1-best)")
    parser.add_argument("--beam-size",
                        type=int,
                        default=8,
                        help="Beam size used during decoding")
    parser.add_argument("--lm",
                        type=str,
                        default="",
                        help="Experimental directory of the RNNLM "
                        "or path of the ngram LM")
    parser.add_argument("--lm-weight",
                        type=float,
                        default=0.1,
                        help="LM score weight used in shallow fusion")
    parser.add_argument("--ctc-weight",
                        type=float,
                        default=0,
                        help="CTC score weight used in beam search")
    parser.add_argument("--lm-tag",
                        type=str,
                        default="best",
                        help="Tag name for RNNLM")
    parser.add_argument("--temperature",
                        type=float,
                        default=1,
                        help="Temperature value used to smooth "
                        "the acoustic scores")
    parser.add_argument("--allow-partial",
                        action=StrToBoolAction,
                        default=True,
                        help="If ture, add partial hypos in the final step")
    parser.add_argument("--len-norm",
                        action=StrToBoolAction,
                        default=False,
                        help="If ture, using length normalized "
                        "when sort nbest hypos")
    parser.add_argument("--len-penalty",
                        type=float,
                        default=0,
                        help="Length penalty factor")
    parser.add_argument("--cov-penalty",
                        type=float,
                        default=0,
                        help="Coverage penalty factor")
    parser.add_argument("--cov-threshold",
                        type=float,
                        default=0.5,
                        help="Coverage threshold value")
    parser.add_argument("--eos-threshold",
                        type=float,
                        default=0,
                        help="Threshold of the EOS symbol")
    parser.add_argument("--show-unk",
                        type=str,
                        default="<unk>",
                        help="Which symbol to show when <unk> exists")
    parser.add_argument("--max-len",
                        type=int,
                        default=1000,
                        help="Maximum decoding sequence length we can have")
    parser.add_argument("--max-len-ratio",
                        type=float,
                        default=1,
                        help="To derive max_len=min(#enc_len*max_len_ratio, "
                        "#max_len)")
    parser.add_argument("--min-len",
                        type=int,
                        default=1,
                        help="Minimum decoding sequence length we can have")
    parser.add_argument("--min-len-ratio",
                        type=float,
                        default=0.0,
                        help="To derive min_len=max(#enc_len*min_len_ratio, "
                        "#min_len)")
    parser.add_argument("--nbest",
                        type=int,
                        default=1,
                        help="N-best decoded utterances to output")
    parser.add_argument("--dump-nbest",
                        type=str,
                        default="",
                        help="If not empty, dump n-best hypothesis")
    parser.add_argument("--dump-align",
                        type=str,
                        default="",
                        help="If assigned, dump alignment "
                        "weight to the directory")
    parser.add_argument("--end-detect",
                        action=StrToBoolAction,
                        default=True,
                        help="If true, detect end of beam search automatically")
    parser.add_argument("--disable-unk",
                        action=StrToBoolAction,
                        default=False,
                        help="If true, disable <unk> symbol "
                        "in decoding sequence")
    return parser


def get_aps_align_parser():
    """
    Return default align parser for aps
    """
    parser = get_asr_base_parser()
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Source scripts of the feature or audio")
    parser.add_argument("text",
                        type=str,
                        help="Text file that we want to get alignment")
    parser.add_argument("alignment",
                        type=str,
                        help="Output file of the alignment")
    parser.add_argument("--word-boundary",
                        type=str,
                        default="",
                        help="To generate the word boundary file if assigned")
    return parser


class BaseTrainParser(object):
    """
    Parser class for training commands
    """
    parser = get_aps_train_parser()


class DecodingParser(object):
    """
    Parser class for decoding commands
    """
    parser = get_aps_decode_parser()


class AlignmentParser(object):
    """
    Parser class for CTC alignment
    """
    parser = get_aps_align_parser()


class DistributedTrainParser(BaseTrainParser):
    """
    Parser class for distributed training
    """
    parser = get_aps_train_parser()
    parser.add_argument("--device-ids",
                        type=str,
                        default="0",
                        help="Training on which GPU devices")
    parser.add_argument("--distributed",
                        type=str,
                        default="none",
                        choices=["torch", "horovod", "none"],
                        help="The distributed backend to use")
    parser.add_argument("--dev-batch-factor",
                        type=int,
                        default=1,
                        help="We will use #batch_size/#dev_batch_factor "
                        "as the batch-size for validation epoch")
