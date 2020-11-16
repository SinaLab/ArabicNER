import torch.nn as nn
from transformers import BertForSequenceClassification


class BertTweetClassifer(nn.Module):
    def __init__(self, num_labels=2):
        super(BertTweetClassifer).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
