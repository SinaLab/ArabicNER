import os
import logging
import argparse
from collections import namedtuple
from arabiner.utils.helpers import load_checkpoint, make_output_dirs, logging_config
from arabiner.utils.data import get_dataloaders, parse_conll_files
from arabiner.utils.metrics import compute_single_label_metrics, compute_multi_label_metrics, compute_nested_metrics

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save results",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path",
    )

    parser.add_argument(
        "--data_paths",
        nargs="+",
        type=str,
        required=True,
        help="Text or sequence to tag, this is in same format as training data with 'O' tag for all tokens",
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
    # Create directory to save predictions
    make_output_dirs(args.output_path, overwrite=True)
    logging_config(log_file=os.path.join(args.output_path, "eval.log"))

    # Load tagger
    tagger, tag_vocab, train_config = load_checkpoint(args.model_path)

    # Convert text to a tagger dataset and index the tokens in args.text
    datasets, vocab = parse_conll_files(args.data_paths)

    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    vocab = vocabs(tokens=vocab.tokens, tags=tag_vocab)

    # From the datasets generate the dataloaders
    dataloaders = get_dataloaders(
        datasets, vocab,
        train_config.data_config,
        batch_size=args.batch_size,
        shuffle=[False] * len(datasets)
    )

    # Evaluate the model on each dataloader
    for dataloader, input_file in zip(dataloaders, args.data_paths):
        filename = os.path.basename(input_file)
        predictions_file = os.path.join(args.output_path, f"predictions_{filename}")
        _, segments, _, _ = tagger.eval(dataloader)
        tagger.segments_to_file(segments, predictions_file)

        if "Nested" in train_config.trainer_config["fn"]:
            compute_nested_metrics(segments, vocab.tags[1:])
        elif "Multi" in train_config.trainer_config["fn"]:
            compute_multi_label_metrics(segments)
        else:
            compute_single_label_metrics(segments)


if __name__ == "__main__":
    main(parse_args())
