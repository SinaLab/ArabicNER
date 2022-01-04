import torch
import torch.nn as nn
from arabiner.nn import BaseModel


class BertNestedTagger(BaseModel):
    def __init__(self, **kwargs):
        super(BertNestedTagger, self).__init__(**kwargs)

        self.max_num_labels = max(self.num_labels)
        classifiers = [nn.Linear(768, num_labels) for num_labels in self.num_labels]
        self.classifiers = torch.nn.Sequential(*classifiers)

    def forward(self, x):
        y = self.bert(x)
        y = self.dropout(y["last_hidden_state"])
        output = list()

        for i, classifier in enumerate(self.classifiers):
            logits = classifier(y)

            # Pad logits to allow Multi-GPU/DataParallel training to work
            # We will truncate the padded dimensions when we compute the loss in the trainer
            logits = torch.nn.ConstantPad1d((0, self.max_num_labels - logits.shape[-1]), 0)(logits)
            output.append(logits)

        # Return tensor of the shape B x T x L x C
        # B: batch size
        # T: sequence length
        # L: number of tag types
        # C: number of classes per tag type
        output = torch.stack(output).permute((1, 2, 0, 3))
        return output

