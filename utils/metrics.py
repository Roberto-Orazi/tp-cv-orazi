import numpy as np


def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))


def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def calculate_metrics_per_class(y_true, y_pred, class_idx):
    tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
    fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
    fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
    tn = np.sum((y_true != class_idx) & (y_pred != class_idx))

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    exactitud = (tp + tn) / (tp + tn + fp + fn)
    f1 = (
        2 * (precision * sensibilidad) / (precision + sensibilidad)
        if (precision + sensibilidad) > 0
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision": precision,
        "exactitud": exactitud,
        "f1": f1,
    }
