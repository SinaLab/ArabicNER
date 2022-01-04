import os
import logging
import json
import argparse
import torch.utils.tensorboard
from torchvision import *
import pickle
from arabiner.utils.data import get_dataloaders, parse_conll_files
from arabiner.utils.helpers import logging_config, load_object, make_output_dirs, set_seed

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
        "--bert_model",
        type=str,
        default="aubmindlab/bert-base-arabertv2",
        help="BERT model",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="GPU IDs to train on",
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
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader number of workers",
    )

    parser.add_argument(
        "--data_config",
        type=json.loads,
        default='{"fn": "arabiner.data.datasets.DefaultDataset", "kwargs": {"max_seq_len": 512}}',
        help="Dataset configurations",
    )

    parser.add_argument(
        "--trainer_config",
        type=json.loads,
        default='{"fn": "arabiner.trainers.BertTrainer", "kwargs": {"max_epochs": 50}}',
        help="Trainer configurations",
    )

    parser.add_argument(
        "--network_config",
        type=json.loads,
        default='{"fn": "arabiner.nn.BertSeqTagger", "kwargs": '
                '{"dropout": 0.1, "bert_model": "aubmindlab/bert-base-arabertv2"}}',
        help="Network configurations",
    )

    parser.add_argument(
        "--optimizer",
        type=json.loads,
        default='{"fn": "torch.optim.AdamW", "kwargs": {"lr": 0.0001}}',
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
        "--seed",
        type=int,
        default=1,
        help="Seed for random initialization",
    )

    args = parser.parse_args()

    return args


def main(args):
    make_output_dirs(
        args.output_path,
        subdirs=("tensorboard", "checkpoints"),
        overwrite=args.overwrite,
    )

    # Set the seed for randomization
    set_seed(args.seed)

    logging_config(os.path.join(args.output_path, "train.log"))
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(args.output_path, "tensorboard")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in args.gpus])

    # Get the datasets and vocab for tags and tokens
    datasets, vocab = parse_conll_files((args.train_path, args.val_path, args.test_path))

    if "Nested" in args.network_config["fn"]:
        args.network_config["kwargs"]["num_labels"] = [len(v) for v in vocab.tags[1:]]
    else:
        args.network_config["kwargs"]["num_labels"] = len(vocab.tags[0])

    # Save tag vocab to desk
    with open(os.path.join(args.output_path, "tag_vocab.pkl"), "wb") as fh:
        pickle.dump(vocab.tags, fh)

    # Write config to file
    args_file = os.path.join(args.output_path, "args.json")
    with open(args_file, "w") as fh:
        logger.info("Writing config to %s", args_file)
        json.dump(args.__dict__, fh, indent=4)

    # From the datasets generate the dataloaders
    args.data_config["kwargs"]["bert_model"] = args.network_config["kwargs"]["bert_model"]
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        datasets, vocab, args.data_config, args.batch_size, args.num_workers
    )

    model = load_object(args.network_config["fn"], args.network_config["kwargs"])
    model = torch.nn.DataParallel(model, device_ids=range(len(args.gpus)))

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

    args.trainer_config["kwargs"].update({
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss": loss,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "log_interval": args.log_interval,
        "summary_writer": summary_writer,
        "output_path": args.output_path
    })

    trainer = load_object(args.trainer_config["fn"], args.trainer_config["kwargs"])
    trainer.train()
    return


if __name__ == "__main__":
    main(parse_args())
