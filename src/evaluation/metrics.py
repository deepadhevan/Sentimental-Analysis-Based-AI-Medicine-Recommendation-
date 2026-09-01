from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
    }

def precision_at_k(relevant: List[str], ranked: List[str], k: int) -> float:
    top = ranked[:k]
    return sum(x in relevant for x in top) / max(k, 1)

def recall_at_k(relevant: List[str], ranked: List[str], k: int) -> float:
    top = ranked[:k]
    return sum(x in relevant for x in top) / max(len(relevant), 1)

def ndcg_at_k(relevant: List[str], ranked: List[str], k: int) -> float:
    top = ranked[:k]
    dcg = sum((1 / np.log2(i + 2)) for i, item in enumerate(top) if item in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_n))
    return float(dcg / idcg) if idcg else 0.0

def mrr(relevant: List[str], ranked: List[str]) -> float:
    for i, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0
