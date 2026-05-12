# 🚀 Opinion Miner Pro — Aspect-Based Sentiment Analysis

## Overview

An end-to-end NLP system that analyses customer reviews, extracts product/service aspects via **spaCy dependency parsing**, and predicts sentiment (positive / negative / neutral) for each aspect using a trained ML model.

**Example:**

> "The camera is amazing but the battery drains quickly."

```json
{
  "aspect_sentiments": {
    "camera": "positive",
    "battery": "negative"
  }
}
```

---

## Architecture

```
opinion-miner-pro/
├── app/
│   ├── backend/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── routes/analyse.py    # /analyse, /bulkAnalyse, /bulkAnalyseResults, /train
│   │   ├── schemas/             # Pydantic request/response models
│   │   ├── services/            # PredictionService, TrainService
│   │   └── deps/auth.py         # API key authentication
│   └── frontend/                # Static HTML/JS/CSS UI
├── src/
│   ├── components/
│   │   ├── prediction.py        # Inference: spaCy aspect extraction + ML sentiment
│   │   ├── data_preprocessing.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   │   └── mongodb_storage.py
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Full training orchestrator
│   │   └── predict_pipeline.py
│   ├── config/
│   │   ├── sentiment_keywords.yaml  # 220+ cue words, weights, thresholds
│   │   └── *.py                     # Dataclass configs
│   └── utils/                   # Logger, exception handler, env loader
├── artifacts/                   # model.pkl, vectorizer.pkl, label_encoder.pkl
├── data/                        # raw.csv, cleaned/, transformed/
├── report/model_scores.yaml     # Training evaluation metrics
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── setup.py
```

---

## Features

- **Aspect extraction** via spaCy dependency parsing (NOUN→ADJ/VERB relations)
- **Sentiment classification** using Logistic Regression + TF-IDF (F1: 0.906)
- **Neutral class** via confidence thresholding (configurable, default < 0.6)
- **220+ sentiment keywords** with calibrated polarity weights
- **Single review** and **bulk CSV** analysis endpoints
- **API key authentication** on all routes
- **MongoDB integration** for data storage and retrieval
- **Static frontend** served from FastAPI
- **Docker-ready** deployment

---

## Tech Stack

- Python 3.11
- scikit-learn (Logistic Regression, LinearSVC, MultinomialNB, SGDClassifier)
- spaCy (dependency parsing, aspect extraction)
- NLTK (tokenization, lemmatization, stopwords)
- FastAPI + Uvicorn
- MongoDB (GridFS for data exchange)
- Pandas / PyArrow

---

## Setup

### Local

```bash
git clone <repo-url>
cd opinion-miner-pro
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

Create a `.env` file:

```
MONGODB_URI=<your-mongodb-connection-string>
API_KEY=<your-api-key>
```

### Docker

```bash
docker build -t opinion-miner .
docker run -p 8000:8000 --env-file .env opinion-miner
```

---

## Running

### Start API server

```bash
uvicorn app.backend.main:app --reload --port 8000
```

The frontend is served at `http://localhost:8000/`.

### Train model

```bash
python src/pipeline/train_pipeline.py
```

Downloads raw data from MongoDB → preprocesses → transforms (TF-IDF) → trains multiple models → saves best to `artifacts/`.

---

## API Endpoints

All endpoints require the `x-api-key` header.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyse` | Analyse a single review |
| `POST` | `/bulkAnalyse` | Upload CSV, get annotated CSV download |
| `POST` | `/bulkAnalyseResults` | Upload CSV, get paginated JSON results |
| `POST` | `/train` | Re-run training pipeline |
| `GET`  | `/health` | Health check |

### POST /analyse

**Request:**
```json
{ "review": "The camera is amazing but the battery drains quickly." }
```

**Response (key fields):**
```json
{
  "overall_sentiment": "positive",
  "confidence": 0.87,
  "aspect_sentiments": { "camera": "positive", "battery": "negative" },
  "main_feature_points": [
    { "feature": "camera", "sentiment": "positive", "evidence": "amazing" },
    { "feature": "battery", "sentiment": "negative", "evidence": "drains" }
  ]
}
```

### POST /bulkAnalyseResults

**Form data:** `file` (CSV), `text_column` (string), `page` (int), `per_page` (int)

### POST /bulkAnalyse

**Form data:** `file` (CSV), `text_column` (string)  
**Returns:** downloadable CSV with sentiment columns appended.

---

## Model Performance

| Model | F1 (weighted) |
|-------|---------------|
| **Logistic Regression** | **0.9059** |
| LinearSVC | 0.9059 |
| SGDClassifier | 0.8862 |
| MultinomialNB | 0.8730 |

Best model: Logistic Regression (C=5, class_weight=balanced)

---

## Configuration

Sentiment keywords, weights, thresholds, and aspect lists are in `src/config/sentiment_keywords.yaml`. Key settings:

- `neutral_confidence_threshold`: 0.6 — predictions below this become "neutral"
- `forced_negative_any`: words that always force negative (e.g., "defective", "scam")
- `preferred_features`: known product aspects prioritized during extraction
- `positive_weights` / `negative_weights`: per-word polarity strength

---

## License

MIT License
