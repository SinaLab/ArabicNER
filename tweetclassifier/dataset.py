import json
import torch
from torchtext.vocab import Vocab
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import itertools
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TweetTransform:
    def __init__(self, bert_model, labels):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        ).encode
        self.label_vocab = Vocab(labels, specials=[])

    def __call__(self, tweet):
        tweet["encoded_seq"] = torch.Tensor(self.tokenizer(tweet["tweet"])).long()
        tweet["encoded_label"] = self.label_vocab.stoi[tweet["labels"][0]]
        return tweet


class TwitterDataset(Dataset):
    def __init__(self, tweets, transform):
        self.transform = transform
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = self.transform(self.tweets[item])
        return tweet["encoded_seq"], tweet["encoded_label"]


def parse_json(data_paths):
    datasets = list()
    labels = list()

    for data_path in data_paths:
        with open(data_path, "r") as fh:
            dataset = json.load(fh)
            datasets.append(dataset)
            labels += itertools.chain(*[tweet["labels"] for tweet in dataset])
            logger.info("%d tweets found in %s", len(dataset), data_path)

    labels = Counter(labels)
    logger.info("Labels %s", labels)

    return datasets, labels


def get_dataloaders(
    datasets, transform, batch_size=32, num_workers=0, shuffle=(True, True, False)
):
    dataloaders = list()

    for i, tweets in enumerate(datasets):
        dataset = TwitterDataset(tweets, transform)

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle[i],
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=batch_collate_fn,
        )

        logger.info("%s batches found", len(dataloader))
        dataloaders.append(dataloader)

    return dataloaders


def batch_collate_fn(batch):
    tweets, labels = zip(*batch)
    tweets = pad_sequence(tweets, batch_first=True, padding_value=0)
    return tweets, torch.Tensor(labels).long()
