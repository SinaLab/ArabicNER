from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from seqeval.scheme import IOB2
from types import SimpleNamespace
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


def compute_nested_metrics(segments, vocabs):
    """
    Compute metrics for multi-class, single label dataset
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    y, y_hat = list(), list()

    for i, vocab in enumerate(vocabs):
        vocab_tags = [tag for tag in vocab.itos if "-" in tag]
        r = re.compile("|".join(vocab_tags))

        y += [[(list(filter(r.match, token.gold_tag)) or ["O"])[0] for token in segment] for segment in segments]
        y_hat += [[token.pred_tag[i]["tag"] for token in segment] for segment in segments]

    logging.info("\n" + classification_report(y, y_hat, scheme=IOB2, digits=4))

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro", scheme=IOB2),
        "macro_f1": f1_score(y, y_hat, average="macro", scheme=IOB2),
        "weights_f1": f1_score(y, y_hat, average="weighted", scheme=IOB2),
        "precision": precision_score(y, y_hat, scheme=IOB2),
        "recall": recall_score(y, y_hat, scheme=IOB2),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)


def compute_multi_label_metrics(segments):
    """
    Compute metrics for multi-class, multi-label dataset
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
                       the metrics are averaged across number of labels
    """
    max_tags = max(len(token.gold_tag) for segment in segments for token in segment)
    y = [[] for _ in range(max_tags)]
    y_hat = [[] for _ in range(max_tags)]
    label_metrics = []

    for i in range(max_tags):
        for tokens in segments:
            segment_gold = []
            segment_pred = []

            for token in tokens:
                preds = [t["tag"] for t in token.pred_tag]
                truth = [t for t in token.gold_tag]
                truth += ["O"] * (max_tags - len(truth))

                segment_gold.append(truth[i])
                segment_pred.append(truth[i] if truth[i] in preds else "O")

            y[i].append(segment_gold)
            y_hat[i].append(segment_pred)

        logging.info("Classification report for entity at position %d", i)
        logging.info("\n" + classification_report(y[i], y_hat[i], scheme=IOB2))

        label_metrics.append(
            {
                "micro_f1": f1_score(y[i], y_hat[i], average="micro", scheme=IOB2),
                "macro_f1": f1_score(y[i], y_hat[i], average="macro", scheme=IOB2),
                "weights_f1": f1_score(y[i], y_hat[i], average="weighted", scheme=IOB2),
                "precision": precision_score(y[i], y_hat[i], scheme=IOB2),
                "recall": recall_score(y[i], y_hat[i], scheme=IOB2),
                "accuracy": accuracy_score(y[i], y_hat[i]),
            }
        )

    # Average metrics across all labels
    metrics = {
        k: np.mean([m[k] for m in label_metrics]) for k in label_metrics[0].keys()
    }

    metrics = SimpleNamespace(**metrics)

    return metrics


def compute_single_label_metrics(segments):
    """
    Compute metrics for multi-class, single label dataset
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    y = [[token.gold_tag[0] for token in segment] for segment in segments]
    y_hat = [[token.pred_tag[0]["tag"] for token in segment] for segment in segments]

    logging.info("\n" + classification_report(y, y_hat, scheme=IOB2))

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro", scheme=IOB2),
        "macro_f1": f1_score(y, y_hat, average="macro", scheme=IOB2),
        "weights_f1": f1_score(y, y_hat, average="weighted", scheme=IOB2),
        "precision": precision_score(y, y_hat, scheme=IOB2),
        "recall": recall_score(y, y_hat, scheme=IOB2),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)
