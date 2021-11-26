import torch.nn as nn
from transformers import BertModel
from arabiner.nn import CRF


class BertCrfSeqTagger(nn.Module):
    def __init__(self, bert_model, num_labels=2, dropout=0.1, start_label=None, stop_label=None):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.decoder = CRF(num_labels, start_label, stop_label)

    def forward(self, x):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        logits = self.linear(y)
        self.decoder(logits)

        return logits
