from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from types import SimpleNamespace


def compute_metrics(golds, preds):
    f1 = f1_score(golds, preds, average="micro")
    precision = precision_score(golds, preds, average="micro")
    recall = recall_score(golds, preds, average="micro")
    accuracy = accuracy_score(golds, preds)

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

    return SimpleNamespace(**metrics)
