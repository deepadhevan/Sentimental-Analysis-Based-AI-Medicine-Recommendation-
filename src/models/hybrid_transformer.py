from typing import List
import torch
from torch import nn
from transformers import AutoModel

class HybridTransformerClassifier(nn.Module):
    """
    Transformer encoder + multi-kernel CNN pooling + aspect embedding.

    The model jointly learns:
      - contextual Transformer representation
      - local n-gram features using 1D convolutions
      - aspect-specific representation
    """
    def __init__(
        self,
        model_name: str,
        num_aspects: int,
        num_sentiments: int = 3,
        dropout: float = 0.2,
        cnn_channels: int = 128,
        kernel_sizes: List[int] = (2, 3, 4),
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, cnn_channels, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.aspect_embedding = nn.Embedding(num_aspects, hidden)
        fused_dim = hidden + cnn_channels * len(kernel_sizes) + hidden

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, num_sentiments)

    def forward(self, input_ids, attention_mask, aspect_ids, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = out.last_hidden_state
        cls = sequence[:, 0, :]

        x = sequence.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feature = torch.relu(conv(x))
            pooled.append(torch.max(feature, dim=2).values)
        cnn_features = torch.cat(pooled, dim=1)

        aspect = self.aspect_embedding(aspect_ids)
        fused = torch.cat([cls, cnn_features, aspect], dim=1)
        logits = self.classifier(self.dropout(fused))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
