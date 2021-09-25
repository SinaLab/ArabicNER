import logging
import argparse
from collections import namedtuple
from arabiner.data.dataset import get_dataloaders, text2segments
from arabiner.data.transforms import BertSeqTransform
from arabiner.utils.helpers import load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    args = parser.parse_args()

    return args


def main(args):
    # Load tagger
    tagger, tag_vocab, train_config = load_checkpoint(args.model_path)

    # Convert text to a tagger dataset and index the tokens in args.text
    dataset, vocab = text2segments(args.text)
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    vocab = vocabs(tags=tag_vocab, tokens=vocab)

    # Load data transformer
    transform = BertSeqTransform(train_config.bert_model, vocab, max_seq_len=train_config.max_seq_len)

    # From the datasets generate the dataloaders
    dataloader = get_dataloaders((dataset,), transform, batch_size=args.batch_size)[0]

    # Perform inference on the text and get back the tagged segments
    segments = tagger.infer(dataloader, vocab=vocab)

    # Print results
    for segment in segments:
        s = [f"{token.text} ({token.pred_tag})" for token in segment]
        print(" ".join(s))

    return


if __name__ == "__main__":
    main(parse_args())
