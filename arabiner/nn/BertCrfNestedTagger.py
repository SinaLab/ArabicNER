import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from arabiner.nn import BaseModel


class BertCrfNestedTagger(BaseModel):
    def __init__(self, **kwargs):
        super(BertCrfNestedTagger, self).__init__(**kwargs)

        crfs = [CRF(num_labels, batch_first=True) for num_labels in self.num_labels]
        self.crfs = torch.nn.Sequential(*crfs)

        classifiers = [nn.Linear(768, num_labels) for num_labels in self.num_labels]
        self.classifiers = torch.nn.Sequential(*classifiers)

    def forward(self, x, labels=None, masks=None):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        losses = list()

        if labels is not None:
            for i, (classifier, crf) in enumerate(zip(self.classifiers, self.crfs)):
                logits = classifier(y)
                loss = -crf(F.log_softmax(logits, 2).long(), labels[:, i, :].long(), reduction='mean', mask=masks[:, i, :].byte())
                losses.append(loss)
            return losses
        else:
            # Final predictions are stored in BATCH_SIZE x SEQ_LEN x TAG_TYPE matrix
            preds = torch.empty((x.shape[0], x.shape[1], len(self.classifiers)))

            for i, (classifier, crf) in enumerate(zip(self.classifiers, self.crfs)):
                logits = classifier(y)
                c_preds = crf.decode(logits)
                preds[:, :, i] = torch.IntTensor(c_preds)

            return preds.to(x.device)

