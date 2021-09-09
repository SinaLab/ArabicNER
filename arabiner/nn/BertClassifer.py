import torch.nn as nn
from transformers import BertModel


class BertClassifer(nn.Module):
    def __init__(self, bert_model, num_labels=2, dropout=0.1):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        _, x = self.bert(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
