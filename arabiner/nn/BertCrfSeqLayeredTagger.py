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
        self.linear2nd = nn.Linear(768 + num_labels, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, labels=None):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        logits = self.linear(y)

        if labels is not None:
            depth = int(labels.shape[0] / x.shape[0])
            labels_at_depth = labels[0::depth]
            loss = -self.crf(F.log_softmax(logits, 2).long(), labels_at_depth.long(), reduction='mean')

            for d in range(1, depth):
                labels_at_depth = labels[d::depth]
                logits = self.linear2nd(torch.cat((y, logits), dim=2))
                loss += -self.crf(F.log_softmax(logits, 2).long(), labels_at_depth.long(), reduction='mean')

            loss /= depth
            return loss
        else:
            prev_preds = None
            preds = list()

            while True:
                depth_preds = self.crf.decode(logits)
                logits = self.linear2nd(torch.cat((y, logits), dim=2))
                all_o = all([_y == 2 for t in depth_preds for _y in t])

                if depth_preds == prev_preds or all_o:
                    break

                preds += depth_preds
                prev_preds = depth_preds

            depth = int(len(preds) / x.shape[0])
            return torch.IntTensor(preds).view(x.shape[0], depth, x.shape[1]).to(x.device)
