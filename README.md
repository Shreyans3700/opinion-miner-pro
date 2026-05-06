# 🚀 Review Intelligence System (Aspect-Based Sentiment Analysis)

## 📌 Overview

The **Review Intelligence System** is an end-to-end NLP project that analyzes customer reviews and extracts **fine-grained insights** by identifying product/service aspects and their corresponding sentiment.

Unlike traditional sentiment analysis (which outputs a single label), this system performs **Aspect-Based Sentiment Analysis (ABSA)** — breaking reviews into multiple aspects and assigning sentiment to each one. ([YouScan][1])

---

## 🎯 Problem Statement

Build a system that:

* Takes raw customer reviews as input
* Identifies key aspects (e.g., *battery, camera, delivery*)
* Predicts sentiment for each aspect (positive/negative)
* Generates actionable insights for businesses

---

## 🧠 Example

**Input:**

> “The camera is amazing but the battery drains quickly.”

**Output:**

```json
{
  "camera": "positive",
  "battery": "negative"
}
```

---

## 🏗️ Project Architecture

```
review-intelligence-system/
│
├── artifacts/
│   ├── model.pkl
│   ├── vectorizer.pkl
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── experimentation.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── aspect_extraction.py
│   │   ├── sentiment_model.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── exception.py
│   │   ├── utils.py
│
├── app/
│   ├── app.py
│   └── streamlit_app.py
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Features

* ✅ Text preprocessing (cleaning, tokenization, lemmatization)
* ✅ Aspect extraction using NLP techniques (POS tagging / TF-IDF)
* ✅ Sentiment classification (Logistic Regression / ML models)
* ✅ Aspect-wise sentiment mapping
* ✅ REST API using FastAPI
* ✅ Interactive UI using Streamlit
* ✅ Logging and exception handling

---

## 🧩 Pipeline

### 1. Data Ingestion

* Load dataset (e.g., product reviews)

### 2. Data Preprocessing

* Clean text
* Remove stopwords
* Lemmatization

### 3. Aspect Extraction

* Extract nouns / key features from text

### 4. Sentiment Classification

* Convert text → TF-IDF vectors
* Train classification model

### 5. Aspect Sentiment Mapping

* Map each aspect to its sentiment

---

## 🚀 Installation

```bash
git clone <repo-url>
cd review-intelligence-system
pip install -r requirements.txt
pip install -e .
```

---

## ▶️ Running the Project

### 🔹 Train Model

```bash
python src/pipeline/train_pipeline.py
```

### 🔹 Run API

```bash
uvicorn app.app:app --reload --port 8000
```

### 🔹 Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## 📡 API Endpoints

### `/predict`

* Input: single review
* Output: sentiment + aspects

### `/analyze_csv`

* Input: CSV file
* Output: aggregated insights

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* Confusion Matrix

---

## 📊 Future Improvements

* 🔥 Replace TF-IDF with Transformer models (BERT)
* 🔥 Improve aspect extraction using NER
* 🔥 Add explainability (LIME / SHAP)
* 🔥 Deploy using Docker / Cloud

---

## 🧠 Tech Stack

* Python
* scikit-learn
* NLP (NLTK / spaCy)
* FastAPI
* Streamlit

---

## 📌 Key Learnings

* End-to-end ML pipeline design
* NLP preprocessing and feature engineering
* Aspect-based sentiment analysis
* Model deployment and API building

---

## 📄 License

MIT License

[1]: https://youscan.io/blog/aspect-based-sentiment-analysis/?utm_source=chatgpt.com "Aspect-Based Sentiment Analysis: The Complete Guide (2026) | YouScan"
