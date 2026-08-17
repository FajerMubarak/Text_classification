# 🍔 Arabic Restaurant Reviews Analysis System

This project is an interactive web application designed to analyze Arabic restaurant reviews. It leverages advanced Natural Language Processing (NLP) models to evaluate customer sentiment and categorize feedback into key operational areas.

---

## 🚀 Live Application

Access the live app here: 👉 (https://textclassification-s.streamlit.app/)

---

## 📝 Project Overview

This application utilizes state-of-the-art Hugging Face Transformer models to evaluate Arabic restaurant feedback. It performs sentiment analysis to capture customer satisfaction and applies zero-shot categorization to organize reviews into operational domains like food quality, service, and pricing.

---

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **AI/ML:** Hugging Face Transformers (`CAMeLBERT`, `mDeBERTa-v3`), PyTorch
* **Data Processing:** Pandas, Scikit-learn, Matplotlib, Seaborn
* **Language:** Python 3.11+

---

## 📌 Features

* **Sentiment Analysis:** Evaluates Arabic review sentiment into **Positive**, **Negative**, or **Neutral** categories using `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment`.
* **Aspect Categorization:** Groups reviews into five operational dimensions (**Food Quality**, **Service & Staff**, **Price**, **Location & Ambience**, **Waiting Time**) using `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.
* **Error Analysis & Performance:** Examines output patterns on dialectal text and multi-topic reviews using confusion matrix heatmaps and confidence score distributions.
* **Interactive Dashboard:** Streamlit interface allowing users to test single review inputs or select pre-loaded samples from `restaurant_reviews.csv` with real-time predictions and JSON metadata logs.
