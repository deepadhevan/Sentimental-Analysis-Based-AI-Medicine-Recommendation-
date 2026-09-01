# Aspect-Aware Hybrid Transformer for AI-Based Medicine Recommendation

Research implementation for:

> **An Aspect-Aware Hybrid Transformer Framework for Sentiment Analysis and Personalized Medicine Recommendation**

## Overview

This repository provides a modular research pipeline for:

1. Text preprocessing
2. Aspect extraction
3. Aspect-level sentiment classification
4. Hybrid Transformer + CNN feature learning
5. Sentiment-aware medicine ranking
6. Evaluation using classification and recommendation metrics

**Important:** This project is for research and experimental purposes only. It does not provide medical advice, diagnosis, prescriptions, or treatment recommendations. Any medicine-related output must be reviewed by a qualified healthcare professional.

## Architecture

```text
Patient/Review Text
       |
       v
Text Preprocessing
       |
       v
Aspect Extraction
       |
       v
Transformer Encoder ----+
       |                |
       +--> CNN --------+--> Hybrid Representation
                              |
                              v
                    Aspect Sentiment Classifier
                              |
                              v
                    Sentiment/Aspect Features
                              |
                              v
                     Medicine Ranking Engine
                              |
                              v
                    Research Recommendation
```

## Dataset format

The training CSV should contain:

```text
text,aspect,sentiment,medicine
"The medicine reduced my pain but caused nausea","effectiveness","positive","Medicine_A"
"The medicine caused nausea","side_effect","negative","Medicine_A"
```

Required columns:

- `text`: patient/review text
- `aspect`: aspect category
- `sentiment`: positive/negative/neutral
- `medicine`: medicine identifier

Do not commit PHI, PII, proprietary clinical data, prescriptions, or confidential datasets.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py \
  --data data/raw/sample_reviews.csv \
  --output models/hybrid_transformer \
  --epochs 3
```

## Evaluation

```bash
python scripts/evaluate.py \
  --data data/raw/sample_reviews.csv \
  --model models/hybrid_transformer
```

## Recommendation demo

```bash
python scripts/recommend.py \
  --text "This medicine helped my headache but caused mild nausea" \
  --catalog data/raw/medicine_catalog.csv \
  --model models/hybrid_transformer
```

## Research metrics

The implementation supports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Recall@K
- Precision@K
- NDCG@K
- MRR

For a publication, also report train/validation/test split strategy, random seeds, baseline models, ablation studies, confidence intervals where appropriate, and dataset provenance.

## Suggested baselines

Compare the proposed model against:

- TF-IDF + Logistic Regression
- BiLSTM
- BERT/DistilBERT
- Transformer without CNN
- Transformer + CNN without aspect features
- Proposed hybrid model

## Reproducibility

The repository fixes random seeds and stores model/training configuration in `configs/`.

## License

MIT License. See `LICENSE`.
