import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import yaml

from src.preprocessing.dataset_preparation import load_dataframe, build_label_maps, split_dataframe
from src.models.hybrid_transformer import HybridTransformerClassifier

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, maps, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.maps = maps
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tokenizer(
            row["text"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "aspect_ids": torch.tensor(self.maps.aspect2id[row["aspect"]], dtype=torch.long),
            "labels": torch.tensor(self.maps.sentiment2id[row["sentiment"]], dtype=torch.long),
        }

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    with open("configs/model_config.yaml") as f:
        mc = yaml.safe_load(f)
    with open("configs/training_config.yaml") as f:
        tc = yaml.safe_load(f)

    seed_everything(tc["seed"])
    df = load_dataframe(args.data)
    maps = build_label_maps(df)
    train_df, val_df, _ = split_dataframe(
        df, tc["seed"], tc["test_size"], tc["validation_size"]
    )

    tokenizer = AutoTokenizer.from_pretrained(mc["model_name"])
    train_ds = ReviewDataset(train_df, tokenizer, maps, mc["max_length"])
    val_ds = ReviewDataset(val_df, tokenizer, maps, mc["max_length"])

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"])

    model = HybridTransformerClassifier(
        mc["model_name"], len(maps.aspect2id), len(maps.sentiment2id),
        mc["dropout"], mc["cnn_channels"], mc["cnn_kernel_sizes"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=mc["learning_rate"], weight_decay=mc["weight_decay"]
    )

    history = []
    for epoch in range(tc["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip_norm"])
            optimizer.step()
            total_loss += out["loss"].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch)["loss"].item()

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
        }
        history.append(row)
        print(row)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    tokenizer.save_pretrained(outdir / "tokenizer")
    (outdir / "label_maps.json").write_text(json.dumps({
        "sentiment2id": maps.sentiment2id,
        "aspect2id": maps.aspect2id,
        "medicine2id": maps.medicine2id,
    }, indent=2))
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved research model to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/hybrid_transformer")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
