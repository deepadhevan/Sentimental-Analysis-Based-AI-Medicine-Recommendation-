from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    dropout: float = 0.2
    cnn_channels: int = 128
    num_sentiments: int = 3
