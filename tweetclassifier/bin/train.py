import argparse
from tweetclassifier.dataset import get_dataloaders
from tweetclassifier.trainer import Trainer
from transformers import BertForSequenceClassification


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

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )

    args = parser.parse_args()

    return args


def main(args):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(input_path=args.input_path)

    trainer = Trainer(model=model,
                      max_epochs=args.max_epochs,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      test_dataloader=test_dataloader)
    trainer.train()
    return


if __name__ == "__main__":
    main(parse_args())
