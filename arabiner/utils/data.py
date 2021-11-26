from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from collections import Counter, namedtuple
import logging
import itertools
from arabiner.utils.helpers import load_object
from arabiner.data.datasets import Token

logger = logging.getLogger(__name__)


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
                parts = token.split()
                token = Token(text=parts[0], gold_tag=parts[1:])
                segment.append(token)

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
        tokens += [token.text for segment in dataset for token in segment]
        tags += [token.gold_tag for segment in dataset for token in segment]

    # Flatten list of tags
    tags = list(itertools.chain(*tags))

    # Generate vocabs for tags and tokens
    vocabs = vocabs(
        tokens=Vocab(Counter(tokens)),
        tags=Vocab(Counter(tags), specials=["<start>", "<stop>", "<unk>", "<pad>"]))
    return tuple(datasets), vocabs


def text2segments(text):
    """
    Convert text to a datasets and index the tokens
    """
    dataset = [[Token(text=token, gold_tag=["O"]) for token in text.split()]]
    tokens = [token.text for segment in dataset for token in segment]

    # Generate vocabs for tags and tokens
    vocab = Vocab(Counter(tokens))
    return dataset, vocab


def get_dataloaders(
    datasets, vocab, data_config, batch_size=32, num_workers=0, shuffle=(True, True, False)
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
        data_config["kwargs"].update({"examples": examples, "vocab": vocab})
        dataset = load_object(data_config["fn"], data_config["kwargs"])

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
