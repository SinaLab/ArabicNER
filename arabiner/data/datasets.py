import logging
import itertools
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from arabiner.data.transforms import (
    BertSeqTransform,
    BertSeqMultiLabelTransform,
    BertSeqLayeredTransform,
    NoneOverlappingSameTypeTransform,
)

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
        self.subwords = None

    @property
    def subwords(self):
        return self._subwords

    @subwords.setter
    def subwords(self, value):
        self._subwords = value

    def __str__(self):
        """
        Token text representation
        :return: str
        """
        gold_tags = "|".join(self.gold_tag)
        pred_tags = "|".join([pred_tag["tag"] for pred_tag in self.pred_tag])

        if self.gold_tag:
            r = f"{self.text}\t{gold_tags}\t{pred_tags}"
        else:
            r = f"{self.text}\t{pred_tags}"

        return r


class DefaultDataset(Dataset):
    def __init__(
        self,
        examples=None,
        vocab=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = BertSeqTransform(bert_model, vocab, max_seq_len=max_seq_len)
        self.examples = examples
        self.vocab = vocab

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
        tags = pad_sequence(
            tags, batch_first=True, padding_value=self.vocab.tags.stoi["O"]
        )
        return subwords, tags, tokens, valid_len


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        examples=None,
        vocab=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = BertSeqMultiLabelTransform(
            bert_model, vocab, max_seq_len=max_seq_len
        )
        self.examples = examples
        self.vocab = vocab

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
        tags = pad_sequence(tags, batch_first=True, padding_value=0)

        return subwords, tags, tokens, valid_len


class LayeredLabelDataset(Dataset):
    def __init__(
        self,
        examples=None,
        vocab=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = BertSeqLayeredTransform(
            bert_model, vocab, max_seq_len=max_seq_len
        )
        self.examples = examples
        self.vocab = vocab

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
        tags = list(tags)
        depth = max([t.shape[0] for t in tags])

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)

        # Pad levels, if the max depth is "depth" then pad all sequences to that depth
        # For example, if the depth is 3 (3 nested entities) and a given sequence in the
        # batch has one level and 50 tokens (1 x 30), then add additional rows to the matrix
        # to become 3 x 50. The additional two rows are all zeros.
        for i in range(len(tags)):
            d = tags[i].shape[0]
            if d < depth:
                pad_level = torch.zeros((depth - d, tags[i].shape[1]))
                tags[i] = torch.cat((tags[i], pad_level))

        # Convert list of lists to a one list
        tags = list(itertools.chain(*tags))

        # Now we will pad the all tags. Note that this will generate one matrix
        # that contains all levels for each all sequences. We can use indexing to
        # find the levels for every sequence
        # i.e. first level for all sequences = tags[0::depth]
        # i.e. second level for all sequences = tags[1::depth]
        # Number of levels in "tags" = tags.shape[0] / subwords.shape[0]
        tags = pad_sequence(
            tags, batch_first=True, padding_value=self.vocab.tags.stoi["O"]
        )

        return subwords, tags, tokens, valid_len


class NoneOverlappingSameTypeDataset(Dataset):
    def __init__(
        self,
        examples=None,
        vocab=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.transform = NoneOverlappingSameTypeTransform(
            bert_model, vocab, max_seq_len=max_seq_len
        )
        self.examples = examples
        self.vocab = vocab

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
        tags = list(tags)
        depth = max([t.shape[0] for t in tags])

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)

        # Pad levels, if the max depth is "depth" then pad all sequences to that depth
        # For example, if the depth is 3 (3 nested entities) and a given sequence in the
        # batch has one level and 50 tokens (1 x 30), then add additional rows to the matrix
        # to become 3 x 50. The additional two rows are all zeros.
        for i in range(len(tags)):
            d = tags[i].shape[0]
            if d < depth:
                pad_level = torch.zeros((depth - d, tags[i].shape[1]))
                tags[i] = torch.cat((tags[i], pad_level))

        # Convert list of lists to a one list
        tags = list(itertools.chain(*tags))

        # Now we will pad the all tags. Note that this will generate one matrix
        # that contains all levels for each all sequences. We can use indexing to
        # find the levels for every sequence
        # i.e. first level for all sequences = tags[0::depth]
        # i.e. second level for all sequences = tags[1::depth]
        # Number of levels in "tags" = tags.shape[0] / subwords.shape[0]
        tags = pad_sequence(
            tags, batch_first=True, padding_value=self.vocab.tags.stoi["O"]
        )

        return subwords, tags, tokens, valid_len
