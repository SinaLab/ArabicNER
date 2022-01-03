import torch
import torch.nn as nn
from arabiner.nn import BaseModel


class BertNestedTagger(BaseModel):
    def __init__(self, **kwargs):
        super(BertNestedTagger, self).__init__(**kwargs)

        classifiers = [nn.Linear(768, num_labels) for num_labels in self.num_labels]
        self.classifiers = torch.nn.Sequential(*classifiers)

    def forward(self, x, labels=None, loss_fn=None):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        preds = torch.empty((x.shape[0], x.shape[1], len(self.classifiers)))
        losses = list()

        if labels is not None:
            for i, classifier in enumerate(self.classifiers):
                logits = classifier(y)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), torch.reshape(labels[:, i, :], (-1,)).long())
                losses.append(loss)
                preds[:, :, i] = torch.argmax(logits, dim=2)

        return preds, losses

