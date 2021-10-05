import os
import sys
import logging
import importlib
import shutil
import torch
import pickle
import json
from argparse import Namespace


def logging_config(log_file=None):
    """
    Initialize custom logger
    :param log_file: str - path to log file, full path
    :return: None
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))
        print("Logging to {}".format(log_file))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        force=True
    )


def load_object(name, kwargs):
    """
    Load objects dynamically given the object name and its arguments
    :param name: str - object name, class name or function name
    :param kwargs: dict - keyword arguments
    :return: object
    """
    object_module, object_name = name.rsplit(".", 1)
    object_module = importlib.import_module(object_module)
    fn = getattr(object_module, object_name)(**kwargs)
    return fn


def make_output_dirs(path, subdirs=[], overwrite=True):
    """
    Create root directory and any other sub-directories
    :param path: str - root directory
    :param subdirs: List[str] - list of sub-directories
    :param overwrite: boolean - to overwrite the directory or not
    :return: None
    """
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)

    os.makedirs(path)

    for subdir in subdirs:
        os.makedirs(os.path.join(path, subdir))


def load_checkpoint(model_path):
    """
    Load model given the model path
    :param model_path: str - path to model
    :return: tagger - arabiner.trainers.BaseTrainer - the tagger model
             tag_vocab - torchtext.vocab.Vocab - indexed tags
             train_config - argparse.Namespace - training configurations
    """
    with open(os.path.join(model_path, "tag_vocab.pkl"), "rb") as fh:
        tag_vocab = pickle.load(fh)

    # Load train configurations from checkpoint
    train_config = Namespace()
    with open(os.path.join(model_path, "args.json"), "r") as fh:
        train_config.__dict__ = json.load(fh)

    # Load BERT tagger
    train_config.network_config["kwargs"]["num_labels"] = len(tag_vocab)
    model = load_object(train_config.network_config["fn"], train_config.network_config["kwargs"])

    if torch.cuda.is_available():
        model = model.cuda()

    # Load the tagger
    train_config.trainer_config["kwargs"]["model"] = model

    tagger = load_object(train_config.trainer_config["fn"], train_config.trainer_config["kwargs"])
    tagger.load(os.path.join(model_path, "checkpoints"))
    return tagger, tag_vocab, train_config
