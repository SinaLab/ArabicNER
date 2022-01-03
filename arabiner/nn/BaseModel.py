from torch import nn
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self,
                 bert_model="aubmindlab/bert-base-arabertv2",
                 num_labels=2,
                 dropout=0.1,
                 num_types=0):
        super().__init__()

        self.bert_model = bert_model
        self.num_labels = num_labels
        self.num_types = num_types
        self.dropout = dropout

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
