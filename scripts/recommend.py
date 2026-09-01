import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.models.hybrid_transformer import HybridTransformerClassifier

def main(args):
    model_dir = Path(args.model)
    saved = json.loads((model_dir / "label_maps.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    model = HybridTransformerClassifier(
        "distilbert-base-uncased",
        len(saved["aspect2id"]),
        len(saved["sentiment2id"])
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    aspect = args.aspect
    aspect_id = saved["aspect2id"].get(aspect, 0)

    enc = tokenizer(
        args.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            aspect_ids=torch.tensor([aspect_id])
        )
        probs = torch.softmax(out["logits"], dim=-1)[0]
        pred_id = int(probs.argmax())

    id2sentiment = {v: k for k, v in saved["sentiment2id"].items()}
    print("Predicted sentiment:", id2sentiment[pred_id])
    print("Confidence:", round(float(probs[pred_id]), 4))

    catalog = pd.read_csv(args.catalog)
    # Research-only ranking: the score is based on model sentiment and catalog aspect coverage.
    rows = []
    for _, r in catalog.iterrows():
        coverage = 1.0 if aspect in str(r["aspects_supported"]).split("|") else 0.0
        score = float(probs[pred_id]) * (0.5 + 0.5 * coverage)
        rows.append((r["medicine"], score))
    rows.sort(key=lambda x: x[1], reverse=True)

    print("\nResearch candidate ranking (NOT a prescription):")
    for med, score in rows:
        print(f"{med}: {score:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--aspect", default="effectiveness")
    p.add_argument("--catalog", default="data/raw/medicine_catalog.csv")
    p.add_argument("--model", required=True)
    main(p.parse_args())
