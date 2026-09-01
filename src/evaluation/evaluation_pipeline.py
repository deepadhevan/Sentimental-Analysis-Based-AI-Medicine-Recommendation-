import torch
from sklearn.metrics import classification_report

@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out["logits"].argmax(dim=-1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
