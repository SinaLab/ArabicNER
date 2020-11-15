import argparse
from tweetclassifier.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to training data",
    )

    args = parser.parse_args()

    return args


def main(args):
    get_dataloaders(input_path=args.input_path)


if __name__ == "__main__":
    main(parse_args())
