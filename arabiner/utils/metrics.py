from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


def compute_metrics(segments):
    y = [[token.gold_tag for token in segment] for segment in segments]
    y_hat = [[token.pred_tag for token in segment] for segment in segments]

    logging.info("\n"+classification_report(y, y_hat))

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro"),
        "macro_f1": f1_score(y, y_hat, average="macro"),
        "weights_f1": f1_score(y, y_hat, average="weighted"),
        "precision": precision_score(y, y_hat),
        "recall": recall_score(y, y_hat),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)
