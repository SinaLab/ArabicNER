import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import itertools
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DefaultDataset(Dataset):
    def __init__(self, examples, transform):
        self.transform = transform
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.transform(self.examples[item])
        return example["encoded_seq"], example["encoded_label"]


def parse_json(data_paths):
    datasets = list()
    labels = list()

    for data_path in data_paths:
        with open(data_path, "r") as fh:
            dataset = json.load(fh)
            datasets.append(dataset)
            labels += itertools.chain(*[tweet["labels"] for tweet in dataset])
            logger.info("%d examples found in %s", len(dataset), data_path)

    labels = Counter(labels)
    logger.info("Labels %s", labels)

    return datasets, labels


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
    text, labels = zip(*batch)
    text = pad_sequence(text, batch_first=True, padding_value=0)
    return text, torch.Tensor(labels).long()
