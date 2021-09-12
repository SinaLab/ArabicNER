import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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


def conll_to_segments(filename):
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token:
                segments.append(segment)
                segment = list()
            else:
                segment.append(token)

        segments.append(segment)

    return segments


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
