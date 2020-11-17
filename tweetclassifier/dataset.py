import json
import torch
from torchtext.vocab import Vocab
from transformers import BertTokenizer
from torch.utils.data import Dataset
import itertools
from collections import Counter


class TweetTransform:
    def __init__(self, bert_model, labels):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True).encode
        self.label_vocab = Vocab(labels)

    def __call__(self, tweet):
        tweet["tokens"] = self.tokenizer(tweet["tweet"])

        return tweet


class TwitterDataset(Dataset):
    def __init__(self, tweets, transform):
        self.transform = transform
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = self.transform(self.tweets[item])
        return tweet


def parse_json(data_paths):
    datasets = list()
    labels = list()

    for data_path in data_paths:
        with open(data_path, "r") as fh:
            dataset = json.load(fh)
            datasets.append(dataset)
            labels += itertools.chain(*[tweet["labels"] for tweet in dataset])

    return datasets, Counter(labels)


def get_dataloaders(datasets, transform):
    for tweets in datasets:
        dataset = TwitterDataset(tweets, transform)
        d = dataset[0]
