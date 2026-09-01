from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from .text_cleaning import clean_text, normalize_aspect

@dataclass
class LabelMaps:
    sentiment2id: Dict[str, int]
    aspect2id: Dict[str, int]
    medicine2id: Dict[str, int]

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "aspect", "sentiment", "medicine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()
    df["text"] = df["text"].map(clean_text)
    df["aspect"] = df["aspect"].map(normalize_aspect)
    return df

def build_label_maps(df: pd.DataFrame) -> LabelMaps:
    def make_map(values: List[str]) -> Dict[str, int]:
        return {v: i for i, v in enumerate(sorted(set(values)))}
    return LabelMaps(
        sentiment2id=make_map(df["sentiment"].tolist()),
        aspect2id=make_map(df["aspect"].tolist()),
        medicine2id=make_map(df["medicine"].tolist()),
    )

def split_dataframe(df: pd.DataFrame, seed=42, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["sentiment"])
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed, stratify=train_val["sentiment"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
