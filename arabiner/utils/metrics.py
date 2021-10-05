from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from types import SimpleNamespace
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    metrics = []

    for i in range(max_tags):
        for tokens in segments:
            segment_gold = []
            segment_pred = []

            for token in tokens:
                preds = [t["tag"] for t in token.pred_tag]
                truth = [t for t in token.gold_tag]

                if len(truth) > i:
                    segment_gold.append(truth[i])
                    segment_pred.append(truth[i] if truth[i] in preds else "O")
                else:
                    segment_gold.append("O")
                    segment_pred.append("O")

            y[i].append(segment_gold)
            y_hat[i].append(segment_pred)

        logging.info("Classification report for entity at position %d", i)
        logging.info(classification_report(y[i], y_hat[i]))

        metrics.append({
            "micro_f1": f1_score(y[i], y_hat[i], average="micro"),
            "macro_f1": f1_score(y[i], y_hat[i], average="macro"),
            "weights_f1": f1_score(y[i], y_hat[i], average="weighted"),
            "precision": precision_score(y[i], y_hat[i]),
            "recall": recall_score(y[i], y_hat[i]),
            "accuracy": accuracy_score(y[i], y_hat[i]),
        })

    agg_metrics = {
        "micro_f1": np.mean([m["micro_f1"] for m in metrics]),
        "macro_f1": np.mean([m["macro_f1"] for m in metrics]),
        "weights_f1": np.mean([m["weights_f1"] for m in metrics]),
        "precision": np.mean([m["precision"] for m in metrics]),
        "recall": np.mean([m["recall"] for m in metrics]),
        "accuracy": np.mean([m["accuracy"] for m in metrics]),
    }

    agg_metrics = SimpleNamespace(**agg_metrics)

    return agg_metrics


def compute_single_label_metrics(segments):
    """
    Compute metrics for multi-class, single label dataset
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    y = [[token.gold_tag[0] for token in segment] for segment in segments]
    y_hat = [[token.pred_tag[0]["tag"] for token in segment] for segment in segments]

    logging.info("\n" + classification_report(y, y_hat))

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro"),
        "macro_f1": f1_score(y, y_hat, average="macro"),
        "weights_f1": f1_score(y, y_hat, average="weighted"),
        "precision": precision_score(y, y_hat),
        "recall": recall_score(y, y_hat),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)


def compute_metrics(segments, multi_label=False):
    """
    Compute metrics on the given segments
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :param multi_label: boolean - True = multi-class/multi-label, False = multi-class/single label
    :return: metric - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    if multi_label:
        metric = compute_multi_label_metrics(segments)
    else:
        metric = compute_single_label_metrics(segments)

    return metric

