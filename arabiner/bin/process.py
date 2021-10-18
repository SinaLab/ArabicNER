import os
import argparse
import csv
import logging
import numpy as np
from arabiner.utils.helpers import logging_config
from arabiner.utils.data import conll_to_segments

logger = logging.getLogger(__name__)


def to_conll_format(input_files, output_path, multi_label=False):
    """
    Parse data files and convert them into CoNLL format
    :param input_files: List[str] - list of filenames
    :param output_path: str - output path
    :param multi_label: boolean - True to process data with mutli-class/multi-label
    :return:
    """
    for input_file in input_files:
        tokens = list()
        prev_sent_id = None

        with open(input_file, "r") as fh:
            r = csv.reader(fh, delimiter="\t", quotechar=" ")
            next(r)

            for row in r:
                sent_id, token, labels = row[1], row[3], row[4].split()
                valid_labels = sum([1 for l in labels if "-" in l or l == "O"]) == len(labels)

                if not valid_labels:
                    logging.warning("Invalid labels found %s", str(row))
                    continue
                if not labels:
                    logging.warning("Token %s has no label", str(row))
                    continue
                if not token:
                    logging.warning("Token %s is missing", str(row))
                    continue
                if len(token.split()) > 1:
                    logging.warning("Token %s has multiple tokens", str(row))
                    continue

                if prev_sent_id is not None and sent_id != prev_sent_id:
                    tokens.append([])

                if multi_label:
                    tokens.append([token] + labels)
                else:
                    tokens.append([token, labels[0]])

                prev_sent_id = sent_id

        num_segments = sum([1 for token in tokens if not token])
        logging.info("Found %d segments and %d tokens in %s", num_segments + 1, len(tokens) - num_segments, input_file)

        filename = os.path.basename(input_file)
        output_file = os.path.join(output_path, filename)

        with open(output_file, "w") as fh:
            fh.write("\n".join(" ".join(token) for token in tokens))
            logging.info("Output file %s", output_file)


def train_dev_test_split(input_files, output_path, train_ratio, dev_ratio):
    segments = list()
    filenames = ["train.txt", "val.txt", "test.txt"]

    for input_file in input_files:
        segments += conll_to_segments(input_file)

    n = len(segments)
    np.random.shuffle(segments)
    datasets = np.split(segments, [int(train_ratio*n), int((train_ratio+dev_ratio)*n)])

    # write data to files
    for i in range(len(datasets)):
        filename = os.path.join(output_path, filenames[i])

        with open(filename, "w") as fh:
            text = "\n\n".join(["\n".join([f"{token.text} {' '.join(token.gold_tag)}" for token in segment]) for segment in datasets[i]])
            fh.write(text)
            logging.info("Output file %s", filename)


def main(args):
    if args.task == "to_conll_format":
        to_conll_format(args.input_files, args.output_path, multi_label=args.multi_label)
    if args.task == "train_dev_test_split":
        train_dev_test_split(args.input_files, args.output_path, args.train_ratio, args.dev_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="List of input files",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        required=False,
        help="Training data ratio (percent of segments). Required with the task train_dev_test_split. "
             "Files must in ConLL format",
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        required=False,
        help="Dev/val data ratio (percent of segments). Required with the task train_dev_test_split. "
             "Files must in ConLL format",
    )

    parser.add_argument(
        "--task", required=True, choices=["to_conll_format", "train_dev_test_split"]
    )

    parser.add_argument(
        "--multi_label", action='store_true'
    )

    args = parser.parse_args()
    logging_config(os.path.join(args.output_path, "process.log"))
    main(args)
