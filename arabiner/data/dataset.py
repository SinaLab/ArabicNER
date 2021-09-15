import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from collections import Counter, namedtuple
import logging

logger = logging.getLogger(__name__)


class Token:
    def __init__(self, text=None, pred_tag=None, gold_tag=None):
        self.text = text
        self.gold_tag = gold_tag
        self.pred_tag = pred_tag

    def __str__(self):
        if self.gold_tag:
            r = f"{self.text}\t{self.gold_tag}\t{self.pred_tag}"
        else:
            r = f"{self.text}\t{self.pred_tag}"

        return r


class DefaultDataset(Dataset):
    def __init__(self, examples, transform):
        self.transform = transform
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        subwords, tags, tokens, valid_len = self.transform(self.examples[item])
        return subwords, tags, tokens, valid_len


def conll_to_segments(filename):
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token:
                segments.append(segment)
                segment = list()
            else:
                segment.append(tuple(token.split()))

        segments.append(segment)

    return segments


def parse_conll_files(data_paths):
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    datasets, tags, tokens = list(), list(), list()

    for data_path in data_paths:
        dataset = conll_to_segments(data_path)
        datasets.append(dataset)
        tokens += [token[0] for segment in dataset for token in segment]
        tags += [token[1] for segment in dataset for token in segment]

    vocabs = vocabs(tokens=Vocab(Counter(tokens)),
                    tags=Vocab(Counter(tags)))
    return tuple(datasets), vocabs


def get_dataloaders(
    datasets, transform, batch_size=32, num_workers=0, shuffle=(True, True, False)
):
    dataloaders = list()

    for i, examples in enumerate(datasets):
        dataset = DefaultDataset(examples, transform)

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
    subwords, tags, tokens, valid_len = zip(*batch)
    subwords = pad_sequence(subwords, batch_first=True, padding_value=0)
    tags = pad_sequence(tags, batch_first=True, padding_value=0)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    return subwords, tags, tokens, valid_len
