from typing import Dict, Iterable, List, Tuple
import math

def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}

def rank_medicines(
    sentiment_probability: float,
    catalog: Iterable[Tuple[str, str]],
) -> List[Tuple[str, float]]:
    """
    Research ranking demo.

    This deliberately treats catalog entries as abstract research candidates.
    It must not be interpreted as a clinical prescription.
    """
    base = {}
    for medicine, aspects in catalog:
        # Simple reproducible score for the research demo.
        supported = len(aspects)
        base[medicine] = sentiment_probability + 0.01 * supported
    probs = softmax(base)
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)
