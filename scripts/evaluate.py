import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.preprocessing.dataset_preparation import load_dataframe, build_label_maps
from src.models.hybrid_transformer import HybridTransformerClassifier
from src.evaluation.metrics import classification_metrics
from scripts.train import ReviewDataset

def main(args):
    model_dir = Path(args.model)
    df = load_dataframe(args.data)
    saved = json.loads((model_dir / "label_maps.json").read_text())

    class Maps: pass
    maps = Maps()
    maps.sentiment2id = saved["sentiment2id"]
    maps.aspect2id = saved["aspect2id"]
    maps.medicine2id = saved["medicine2id"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    ds = ReviewDataset(df, tokenizer, maps, 128)
    loader = DataLoader(ds, batch_size=16)

    model = HybridTransformerClassifier(
        "distilbert-base-uncased",
        len(maps.aspect2id),
        len(maps.sentiment2id)
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(out["logits"].argmax(dim=-1).tolist())

    metrics = classification_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/metrics/classification_metrics.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    main(p.parse_args())
