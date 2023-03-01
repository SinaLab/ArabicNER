from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from seqeval.scheme import IOB2
from types import SimpleNamespace
import logging
import re

logger = logging.getLogger(__name__)


def compute_nested_metrics(segments, vocabs):
    """
    Compute metrics for nested NER
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    y, y_hat = list(), list()

    # We duplicate the dataset N times, where N is the number of entity types
    # For each copy, we create y and y_hat
    # Example: first copy, will create pairs of ground truth and predicted labels for entity type GPE
    #          another copy will create pairs for LOC, etc.
    for i, vocab in enumerate(vocabs):
        vocab_tags = [tag for tag in vocab.get_itos() if "-" in tag]
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


def compute_single_label_metrics(segments):
    """
    Compute metrics for flat NER
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
