import os
import logging
import json
import argparse
import torch.utils.tensorboard
from torchvision import *
import pickle
from arabiner.trainers import BertTrainer
from arabiner.data.dataset import get_dataloaders, parse_conll_files
from arabiner.data.transforms import BertSeqTransform
from arabiner.utils.helpers import logging_config, load_object, make_output_dirs
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
        default="aubmindlab/bert-base-arabertv2",
        help="BERT model",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log results every that many timesteps",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--optimizer",
        type=json.loads,
        default='{"fn": "torch.optim.Adam", "kwargs": {"lr": 0.0001}}',
        help="Optimizer configurations",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=json.loads,
        default='{"fn": "torch.optim.lr_scheduler.ExponentialLR", "kwargs": {"gamma": 1}}',
        help="Learning rate scheduler configurations",
    )

    parser.add_argument(
        "--loss",
        type=json.loads,
        default='{"fn": "torch.nn.CrossEntropyLoss", "kwargs": {}}',
        help="Loss function configurations",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory",
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    return args


def main(args):
    make_output_dirs(
        args.output_path,
        subdirs=("tensorboard", "checkpoints"),
        overwrite=args.overwrite,
    )
    logging_config(os.path.join(args.output_path, "train.log"))
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(args.output_path, "tensorboard")
    )
    args_file = os.path.join(args.output_path, "args.json")

    with open(args_file, "w") as fh:
        logger.info("Writing config to %s", args_file)
        json.dump(args.__dict__, fh, indent=4)

    # Get the datasets and vocab for tags and tokens
    datasets, vocab = parse_conll_files((args.train_path, args.val_path, args.test_path))

    # Save tag vocab to desk
    with open(os.path.join(args.output_path, "tag_vocab.pkl"), "wb") as fh:
        pickle.dump(vocab.tags, fh)

    transform = BertSeqTransform(args.bert_model, vocab, max_seq_len=args.max_seq_len)

    # From the datasets generate the dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        datasets, transform, batch_size=args.batch_size
    )

    # Load BERT tagger
    model = BertSeqTagger(args.bert_model, num_labels=len(vocab.tags), dropout=0.1)

    if torch.cuda.is_available():
        model = model.cuda()

    args.optimizer["kwargs"]["params"] = model.parameters()
    optimizer = load_object(args.optimizer["fn"], args.optimizer["kwargs"])

    args.lr_scheduler["kwargs"]["optimizer"] = optimizer
    if "num_training_steps" in args.lr_scheduler["kwargs"]:
        args.lr_scheduler["kwargs"]["num_training_steps"] = args.max_epochs * len(
            train_dataloader
        )

    scheduler = load_object(args.lr_scheduler["fn"], args.lr_scheduler["kwargs"])
    loss = load_object(args.loss["fn"], args.loss["kwargs"])

    trainer = BertTrainer(
        model=model,
        max_epochs=args.max_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        log_interval=args.log_interval,
        summary_writer=summary_writer,
        output_path=args.output_path,
        vocab=vocab
    )
    trainer.train()
    return


if __name__ == "__main__":
    main(parse_args())
