import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertModel


class BertCrfSeqLayeredTagger(nn.Module):
    def __init__(self, bert_model, num_labels=2, dropout=0.1):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, labels=None):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        logits = self.linear(y)

        # If we have labels, then compute the  log likelihood
        # otherwise, find the best tag sequence
        if labels is not None:
            loss = -self.crf(F.log_softmax(logits, 2), labels, reduction='mean')
            return loss
        else:
            preds = self.crf.decode(logits)
            return torch.IntTensor(preds).to(x.device)
