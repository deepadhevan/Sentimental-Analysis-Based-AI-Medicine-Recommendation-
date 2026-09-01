from typing import Dict, List
import torch
from transformers import AutoTokenizer
from src.models.hybrid_transformer import HybridTransformerClassifier

class MedicineRecommender:
    def __init__(self, model, tokenizer, aspect2id: Dict[str, int], id2sentiment: Dict[int, str]):
        self.model = model
        self.tokenizer = tokenizer
        self.aspect2id = aspect2id
        self.id2sentiment = id2sentiment

    @torch.no_grad()
    def predict_sentiment(self, text: str, aspect: str):
        self.model.eval()
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        aspect_id = torch.tensor([self.aspect2id.get(aspect, 0)])
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            aspect_ids=aspect_id,
        )
        probabilities = torch.softmax(out["logits"], dim=-1)[0]
        idx = int(torch.argmax(probabilities))
        return self.id2sentiment[idx], float(probabilities[idx])
