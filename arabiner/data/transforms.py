import torch
from transformers import BertTokenizer
from functools import partial
import logging

logger = logging.getLogger(__name__)


class BertSeqTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.vocab = vocab

    def __call__(self, segment):
        subwords, tags, tokens = list(), list(), list()

        for token, tag in segment:
            token_subwords = self.encoder(token)[1:-1]
            subwords += token_subwords
            tags += [self.vocab.tags.stoi[tag]] + [self.vocab.tags.stoi["O"]] * (len(token_subwords) - 1)
            tokens += [self.vocab.tokens.stoi[token]] + [self.vocab.tokens.stoi["UNK"]] * (len(token_subwords) - 1)

        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)

        tokens.insert(0, self.tokenizer.cls_token_id)
        tokens.append(self.tokenizer.sep_token_id)

        tags.insert(0, self.vocab.tags.stoi["O"])
        tags.append(self.vocab.tags.stoi["O"])

        return torch.LongTensor(subwords), torch.LongTensor(tags), torch.LongTensor(tokens), len(tokens)
