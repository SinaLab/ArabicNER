import torch
from transformers import BertTokenizer
from functools import partial
import logging
import arabiner

logger = logging.getLogger(__name__)


class BertSeqTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab

    def __call__(self, segment):
        subwords, tags, tokens = list(), list(), list()
        unk_token = arabiner.data.datasets.Token(text="UNK")

        for token in segment:
            token_subwords = self.encoder(token.text)[1:-1]
            subwords += token_subwords
            tags += [self.vocab.tags.stoi[token.gold_tag[0]]] + [self.vocab.tags.stoi["O"]] * (len(token_subwords) - 1)
            tokens += [token] + [unk_token] * (len(token_subwords) - 1)

        # Truncate to max_seq_len
        if len(subwords) > self.max_seq_len - 2:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            logger.info("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            subwords = subwords[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]
            tokens = tokens[:self.max_seq_len - 2]

        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)

        tags.insert(0, self.vocab.tags.stoi["O"])
        tags.append(self.vocab.tags.stoi["O"])

        tokens.insert(0, unk_token)
        tokens.append(unk_token)

        return torch.LongTensor(subwords), torch.LongTensor(tags), tokens, len(tokens)


class BertSeqMultiLabelTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab

    def one_hot(self, tags):
        labels = [self.vocab.tags.stoi[tag] for tag in tags]
        labels = torch.tensor(labels).unsqueeze(0)
        y_onehot = torch.zeros(labels.size(0), len(self.vocab.tags)).scatter_(1, labels, 1.)
        return y_onehot

    def __call__(self, segment):
        subwords, tags, tokens = list(), list(), list()
        unk_token = arabiner.data.datasets.Token(text="UNK")

        for token in segment:
            token_subwords = self.encoder(token.text)[1:-1]
            subwords += token_subwords

            # List of tags for the first subword
            tags.append(self.one_hot(token.gold_tag))

            # List of "O" tags for the remaining subwords
            tags += [self.one_hot(["O"])] * (len(token_subwords) - 1)

            tokens += [token] + [unk_token] * (len(token_subwords) - 1)

        # Truncate to max_seq_len
        if len(subwords) > self.max_seq_len - 2:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            logger.info("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            subwords = subwords[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]
            tokens = tokens[:self.max_seq_len - 2]

        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)

        tags.insert(0, self.one_hot(["O"]))
        tags.append(self.one_hot(["O"]))

        tokens.insert(0, unk_token)
        tokens.append(unk_token)

        return torch.LongTensor(subwords), torch.stack(tags).squeeze(1), tokens, len(tokens)
