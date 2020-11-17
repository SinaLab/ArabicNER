import argparse
from tweetclassifier.dataset import get_dataloaders
from tweetclassifier.trainer import Trainer
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from tweetclassifier.dataset import TweetTransform, parse_json


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training data",
    )

    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to training data",
    )

    parser.add_argument(
        "--test_path",
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

    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="BERT model",
    )

    args = parser.parse_args()

    return args


def main(args):
    datasets, labels = parse_json((args.train_path, args.val_path, args.test_path))
    transform = TweetTransform(args.bert_model, labels)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(datasets, transform)

    model = BertForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=len(label_field.vocab.itos),
        output_attentions=False,
        output_hidden_states=False,
    )

    trainer = Trainer(model=model,
                      max_epochs=args.max_epochs,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      test_dataloader=test_dataloader)
    trainer.train()
    return


if __name__ == "__main__":
    main(parse_args())
