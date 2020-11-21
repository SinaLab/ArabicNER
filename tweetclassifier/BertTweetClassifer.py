import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertTweetClassifer(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super().__init__()

        self.encoder = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x[1])
        y = self.classifier(x)
        return y
