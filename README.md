# 🍔 Arabic Restaurant Reviews Analysis System

An end-to-end Natural Language Processing (NLP) system designed to analyze and classify Arabic restaurant reviews. The application utilizes state-of-the-art Hugging Face Transformer models for both fine-grained sentiment analysis and zero-shot aspect classification, accessible via an interactive Streamlit web dashboard.

---

## 📌 Features

* **Task A: Sentiment Analysis:** Classifies user review sentiment into **Positive**, **Negative**, or **Neutral** categories using a specialized Arabic BERT backbone (`CAMeLBERT-mix`).
* **Task B: Zero-Shot Aspect Classification:** Automatically categorizes reviews into predefined operational dimensions (**Food Quality**, **Service & Staff**, **Price**, **Location & Ambience**, **Waiting Time**) using `mDeBERTa-v3`.
* **Error Analysis & Performance Metrics:** Evaluates model outputs using confusion matrices, confidence score distributions, and classification reports.
* **Interactive Streamlit App:** Allows users to test single review inputs or select pre-loaded reviews from the dataset with real-time predictions and custom JSON metadata outputs.

---

## 📂 Repository Structure

```text
├── data/
│   └── restaurant_reviews.csv      # Dataset containing raw Arabic reviews & ground truth labels
├── notebooks/
│   └── review_analysis.ipynb      # Jupyter notebook for EDA, model evaluation, and error analysis
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Dependencies required for deployment
├── .gitignore                     # Git ignore file for environment & cache files
└── README.md                      # Project documentation
