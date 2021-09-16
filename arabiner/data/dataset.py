from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from collections import Counter, namedtuple
import logging

logger = logging.getLogger(__name__)


class Token:
    def __init__(self, text=None, pred_tag=None, gold_tag=None):
        """
        Token object to hold token attributes
        :param text: str
        :param pred_tag: str
        :param gold_tag: str
        """
        self.text = text
        self.gold_tag = gold_tag
        self.pred_tag = pred_tag

    def __str__(self):
        """
        Token text represenation
        :return: str
        """
        if self.gold_tag:
            r = f"{self.text}\t{self.gold_tag}\t{self.pred_tag}"
        else:
            r = f"{self.text}\t{self.pred_tag}"

        return r


class DefaultDataset(Dataset):
    def __init__(self, examples, transform):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param transform: torchvision.transforms.Compose
        """
        self.transform = transform
        self.examples = examples
        self.vocab = transform.transforms[0].vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        subwords, tags, tokens, valid_len = self.transform(self.examples[item])
        return subwords, tags, tokens, valid_len

    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        subwords, tags, tokens, valid_len = zip(*batch)

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)
        tags = pad_sequence(tags, batch_first=True, padding_value=self.vocab.tags.stoi["O"])
        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        return subwords, tags, tokens, valid_len


def conll_to_segments(filename):
    """
    Convert CoNLL files to segments. This return list of segments and each segment is
    a list of tuples (token, tag)
    :param filename: Path
    :return: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
    """
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
    """
    Parse CoNLL formatted files and return list of segments for each file and index
    the vocabs and tags across all data_paths
    :param data_paths: tuple(Path) - tuple of filenames
    :return: tuple( [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i]
                    [[(token, tag), ...], [(token, tag), ...]], -> segments for data_paths[i+1],
                    ...
                  )
             List of segments for each dataset and each segment has list of (tokens, tags)
    """
    vocabs = namedtuple("Vocab", ["tags", "tokens"])
    datasets, tags, tokens = list(), list(), list()

    for data_path in data_paths:
        dataset = conll_to_segments(data_path)
        datasets.append(dataset)
        tokens += [token[0] for segment in dataset for token in segment]
        tags += [token[1] for segment in dataset for token in segment]

    # Generate vocabs for tags and tokens
    vocabs = vocabs(tokens=Vocab(Counter(tokens)),
                    tags=Vocab(Counter(tags)))
    return tuple(datasets), vocabs


def get_dataloaders(
    datasets, transform, batch_size=32, num_workers=0, shuffle=(True, True, False)
):
    """
    From the datasets generate the dataloaders
    :param datasets: list - list of the datasets, list of list of segments and tokens
    :param transform: torchvision.transforms.Compose
    :param batch_size: int
    :param num_workers: int
    :param shuffle: boolean - to shuffle the data or not
    :return: List[torch.utils.data.DataLoader]
    """
    dataloaders = list()

    for i, examples in enumerate(datasets):
        dataset = DefaultDataset(examples, transform)

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle[i],
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )

        logger.info("%s batches found", len(dataloader))
        dataloaders.append(dataloader)

    return dataloaders
