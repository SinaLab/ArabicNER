import logging
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from arabiner.data.transforms import (
    BertSeqTransform,
    NestedTagsTransform
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
            tags, batch_first=True, padding_value=self.vocab.tags[0].stoi["O"]
        )
        return subwords, tags, tokens, valid_len


class NestedTagsDataset(Dataset):
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
        self.transform = NestedTagsTransform(
            bert_model, vocab, max_seq_len=max_seq_len
        )
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        subwords, tags, tokens, masks, valid_len = self.transform(self.examples[item])
        return subwords, tags, tokens, masks, valid_len

    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        subwords, tags, tokens, masks, valid_len = zip(*batch)

        # Pad sequences in this batch
        # subwords and tokens are padded with zeros
        # tags are padding with the index of the O tag
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)

        masks = [torch.nn.ConstantPad1d((0, subwords.shape[-1] - tag.shape[-1]), 0)(mask)
                 for tag, mask in zip(tags, masks)]
        masks = torch.cat(masks)

        # Pad the tags, do the padding for each tag type
        tags = [torch.nn.ConstantPad1d((0, subwords.shape[-1] - tag.shape[-1]), vocab.stoi["<pad>"])(tag)
                for tag, vocab in zip(tags, self.vocab.tags[1:])]
        tags = torch.cat(tags)

        return subwords, tags, tokens, masks, valid_len
