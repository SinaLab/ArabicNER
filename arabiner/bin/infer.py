import logging
import argparse
from collections import namedtuple
from arabiner.utils.helpers import load_checkpoint
from arabiner.utils.data import get_dataloaders, text2segments

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
    dataset, token_vocab = text2segments(args.text)

    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    vocab = vocabs(tokens=token_vocab, tags=tag_vocab)

    # From the datasets generate the dataloaders
    dataloader = get_dataloaders(
        (dataset,),
        vocab,
        train_config.data_config,
        batch_size=args.batch_size,
        shuffle=(False,),
    )[0]

    # Perform inference on the text and get back the tagged segments
    segments = tagger.infer(dataloader)

    # Print results
    for segment in segments:
        s = [
            f"{token.text} ({'|'.join([t['tag'] for t in token.pred_tag])})"
            for token in segment
        ]
        print(" ".join(s))


if __name__ == "__main__":
    main(parse_args())
