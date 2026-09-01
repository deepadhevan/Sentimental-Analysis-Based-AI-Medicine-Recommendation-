from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr

def test_recommendation_metrics():
    relevant = ["A", "B"]
    ranked = ["A", "C", "B"]
    assert precision_at_k(relevant, ranked, 2) == 0.5
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert ndcg_at_k(relevant, ranked, 2) > 0
    assert mrr(relevant, ranked) == 1.0
