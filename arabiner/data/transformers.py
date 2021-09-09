import torch
from torchtext.vocab import Vocab
from transformers import BertTokenizer
from functools import partial
import logging

logger = logging.getLogger(__name__)


class BertSeqTransform:
    def __init__(self, bert_model, max_seq_len=512):
        self.tokenizer = partial(
            BertTokenizer.from_pretrained(bert_model, do_lower_case=True).encode,
            max_length=max_seq_len,
            truncation=True,
        )

    def __call__(self, example):
        example["encoded_seq"] = torch.Tensor(self.tokenizer(example["text"])).long()
        return example


class LabelTransform:
    def __init__(self, labels, multi_label=False):
        self.label_vocab = Vocab(labels, specials=[])
        self.multi_label = multi_label

    def __call__(self, example):
        example["encoded_label"] = self.label_vocab.stoi[example["labels"][0]]
        return example