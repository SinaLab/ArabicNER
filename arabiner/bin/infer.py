import os
import logging
import json
import argparse
import torch.utils.tensorboard
from torchvision import *
import pickle
from collections import namedtuple
from arabiner.trainers import BertTrainer
from arabiner.data.dataset import get_dataloaders, text2segments
from arabiner.data.transforms import BertSeqTransform
from arabiner.utils.helpers import logging_config, make_output_dirs
from arabiner.nn.BertSeqTagger import BertSeqTagger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path",
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text or sequence to tag",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory",
    )

    args = parser.parse_args()

    return args


def main(args):
    make_output_dirs(
        args.output_path,
        overwrite=args.overwrite,
    )

    logging_config(os.path.join(args.output_path, "train.log"))
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(args.output_path, "tensorboard")
    )
    args_file = os.path.join(args.output_path, "args.json")

    # Save infer configs to file
    with open(args_file, "w") as fh:
        logger.info("Writing config to %s", args_file)
        json.dump(args.__dict__, fh, indent=4)

    # Load tag vocabs generated from the trained model
    with open(os.path.join(args.model_path, "tag_vocab.pkl"), "rb") as fh:
        tag_vocab = pickle.load(fh)

    # Load train configurations from checkpoint
    train_config = argparse.Namespace()
    with open(os.path.join(args.model_path, "args.json"), "r") as fh:
        train_config.__dict__ = json.load(fh)

    # Convert text to a tagger dataset and index the tokens in args.text
    dataset, vocab = text2segments(args.text)
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    vocab = vocabs(tags=tag_vocab, tokens=vocab)

    # Load data transformer
    transform = transforms.Compose(
        [
            BertSeqTransform(train_config.bert_model, vocab, max_seq_len=args.max_seq_len)
        ]
    )

    # From the datasets generate the dataloaders
    dataloader = get_dataloaders((dataset,), transform, batch_size=args.batch_size)[0]

    # Load BERT tagger
    model = BertSeqTagger(train_config.bert_model, num_labels=len(vocab.tags), dropout=0)

    if torch.cuda.is_available():
        model = model.cuda()

    # Load the tagger
    tagger = BertTrainer(
        model=model,
        output_path=args.output_path,
        vocab=vocab,
        summary_writer=summary_writer
    )
    tagger.load(os.path.join(args.model_path, "checkpoints"))

    # Perform inference on the text and get back the preds (predicted tags) and the index tokens
    # so we can reconstruct the original text
    golds, preds, tokens, valid_lens = tagger.infer(dataloader)

    # Convert the output of tagger.segment() to segments
    segments = tagger.to_segments(golds, preds, tokens, valid_lens)

    # Print results
    for segment in segments:
        s = [f"{token.text} ({token.pred_tag})" for token in segment]
        logger.info(" ".join(s))

    return


if __name__ == "__main__":
    main(parse_args())
