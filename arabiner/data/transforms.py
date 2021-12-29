import torch
from transformers import BertTokenizer
from functools import partial
import logging
import re
import itertools
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


class BertSeqLayeredTransform:
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
        tags, tokens, token2subwords = list(), list(), list()
        unk_token = arabiner.data.datasets.Token(text="UNK")

        # Number of nested tag levels is defined by the token with most tags
        num_levels = max([len(token.gold_tag) for token in segment])

        for token in segment:
            token2subwords.append(self.encoder(token.text)[1:-1])

            # Add UNK token for all subwords except for the first subword
            tokens += [token] + [unk_token] * (len(token2subwords[-1]) - 1)

        subwords = list(itertools.chain(*token2subwords))

        for l in range(num_levels):
            level_tags = list()
            for t, token in enumerate(segment):
                # Get the tags for this token at al levels, some levels do not exist for this token, in
                # such case we will use the "O" tag
                token_tags = token.gold_tag + ["O"] * (num_levels - len(token.gold_tag))

                # convert tags to indices and add "O" to all subwords except for the the first subword
                level_tags += [self.vocab.tags.stoi[token_tags[l]]] + [self.vocab.tags.stoi["O"]] * (len(token2subwords[t]) - 1)

            tags.append(level_tags)

        # Truncate to max_seq_len
        if len(subwords) > self.max_seq_len - 2:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            logger.info("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            subwords = subwords[:self.max_seq_len - 2]
            tags = [t[:self.max_seq_len - 2] for t in tags]
            tokens = tokens[:self.max_seq_len - 2]

        # Add "O" tags for the first and last subwords
        tags = torch.Tensor(tags)
        tags = torch.column_stack((
            torch.full((num_levels,), self.vocab.tags.stoi["O"]),
            tags,
            torch.full((num_levels,), self.vocab.tags.stoi["O"]),
        ))

        tokens.insert(0, unk_token)
        tokens.append(unk_token)

        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)
        subwords = torch.LongTensor(subwords)

        return subwords, tags, tokens, len(tokens)


class NoneOverlappingSameTypeTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.tag_names = list(set([tag.split("-", 1)[1] for tag in vocab.tags.itos if "-" in tag]))

    def __call__(self, segment):
        tags, tokens, subwords = list(), list(), list()
        unk_token = arabiner.data.datasets.Token(text="UNK")

        # Encode each token and get its subwords and IDs
        for token in segment:
            token.subwords = self.encoder(token.text)[1:-1]
            subwords += token.subwords
            tokens += [token] + [unk_token] * (len(token.subwords ) - 1)

        # Construct the labels for each tag type
        # The sequence will have a list of tags for each type
        # The final tags for a sequence is a matrix NUM_TAG_TYPES x SEQ_LEN
        # Example:
        #   [
        #       [O, O, B-PERS, I-PERS, O, O, O]
        #       [B-ORG, I-ORG, O, O, O, O, O]
        #       [O, B-GPE, O, O, O, O, O]
        #   ]
        for tag_name in self.tag_names:
            r = re.compile(".*-" + tag_name)
            single_type_tags = [(list(filter(r.match, token.gold_tag))
                                or ["O"]) + ["O"] * (len(token.subwords) - 1)
                                for token in segment]
            single_type_tags = list(itertools.chain(*single_type_tags))
            tags.append([self.vocab.tags.stoi[tag] for tag in single_type_tags])

        # Truncate to max_seq_len
        if len(subwords) > self.max_seq_len - 2:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            logger.info("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            subwords = subwords[:self.max_seq_len - 2]
            tags = [t[:self.max_seq_len - 2] for t in tags]
            tokens = tokens[:self.max_seq_len - 2]

        # Add dummy token at the start end of sequence
        tokens.insert(0, unk_token)
        tokens.append(unk_token)

        # Add CLS and SEP at start end of subwords
        subwords.insert(0, self.tokenizer.cls_token_id)
        subwords.append(self.tokenizer.sep_token_id)
        subwords = torch.LongTensor(subwords)

        # Add "O" tags for the first and last subwords
        tags = torch.Tensor(tags)
        tags = torch.column_stack((
            torch.full((len(self.tag_names),), self.vocab.tags.stoi["O"]),
            tags,
            torch.full((len(self.tag_names),), self.vocab.tags.stoi["O"]),
        ))

        return subwords, tags, tokens, len(tokens)

