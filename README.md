# Sentiment Analysis of Amazon Kindle Reviews
### Comparing Traditional ML vs Deep Learning for NLP

> **Applied Artificial Intelligence | University of Hull**  
> Author: Muhammad Salahuddin

---

## 📌 Project Overview

This project builds and compares **five machine learning models** for binary sentiment classification on Amazon Kindle product reviews — determining whether a review is **Positive** or **Negative** based on its text.

The core question: *Do deep learning models outperform traditional ML for sentiment analysis, or does simplicity win?*

**Spoiler: Naive Bayes beat the neural networks.**

---

## 📊 Results Summary

| Model | Accuracy | ROC-AUC | PR-AUC |
|-------|----------|---------|--------|
| **Naive Bayes** ⭐ | **84.06%** | — | — |
| SVM | ~83% | — | — |
| Simple RNN | 83.58% | — | — |
| LSTM | 81.4% | — | — |
| **CNN** | — | **90.01%** | **94.54%** |

> **Key Finding:** Naive Bayes achieved the highest overall accuracy, while CNN led on ROC-AUC and PR-AUC. Simple RNN outperformed LSTM, suggesting that added complexity does not always improve performance on short-form text.

All five models exceeded the **majority-class baseline**, confirming meaningful learning across all approaches.

---

## 🗂️ Dataset

- **Source:** Amazon Kindle Store Reviews (publicly available)
- **Task:** Binary sentiment classification — Positive (3–5 stars) vs Negative (1–2 stars)
- **Class Imbalance:** Addressed using **SMOTE** (Synthetic Minority Oversampling Technique) on the training set
- **Preprocessing:** Lowercasing, HTML stripping, punctuation removal, tokenisation with Gensim, bigram detection, stopword removal

---

## 🛠️ Models Implemented

### Traditional Machine Learning
| Model | Key Tuning |
|-------|-----------|
| **Multinomial Naive Bayes** | GridSearchCV over `alpha`, `fit_prior` |
| **Support Vector Machine (SVM)** | GridSearchCV over `C`, linear kernel |

### Deep Learning (Keras / TensorFlow)
| Model | Architecture |
|-------|-------------|
| **LSTM** | Embedding → SpatialDropout → LSTM → GlobalAvgPool → Dense |
| **Simple RNN** | Embedding → SpatialDropout → SimpleRNN → GlobalAvgPool → Dense |
| **CNN** | Embedding → SpatialDropout → Conv1D → GlobalMaxPool → Dense |

All deep learning models used:
- Class weighting to handle imbalance
- Early stopping (patience=3) to prevent overfitting
- Before/after hyperparameter tuning comparison

---

## 🔬 Evaluation Metrics

Each model was evaluated on:
- **Accuracy** — overall correctness
- **Precision / Recall / F1-Score** — per-class performance
- **ROC-AUC** — discrimination ability across thresholds
- **PR-AUC** — performance under class imbalance
- **Confusion Matrix** — error analysis
- **Training/Validation curves** — overfitting diagnosis

---

## 📈 Key Findings

1. **Traditional ML generalised better** on this dataset — Naive Bayes at 84.06% outperformed all three deep learning models on raw accuracy
2. **CNN was the strongest deep learning model**, achieving the highest ROC-AUC (90.01%) and PR-AUC (94.54%), making it the best choice when false negatives are costly
3. **Simple RNN > LSTM** (83.58% vs 81.4%) — Kindle reviews are short enough that LSTM's long-range memory provides no benefit
4. **SMOTE + class weighting** were critical — without balancing, models defaulted to always predicting Positive

---

## 🧪 Preprocessing Pipeline

```
Raw Text
  → Clean (lowercase, remove HTML/punctuation)
  → Tokenise (Gensim simple_preprocess)
  → Remove stopwords (+ domain-specific: 'book', 'read')
  → Build bigrams
  → Vectorise (CountVectorizer for ML / Keras Tokenizer + padding for DL)
  → SMOTE balancing (Traditional ML) / Class weights (Deep Learning)
```

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Gensim
- **NLP** — CountVectorizer, TF-IDF, Tokenizer, Bigrams, Stopword removal
- **ML Models** — MultinomialNB, SVC, LSTM, SimpleRNN, Conv1D
- **Evaluation** — Sklearn metrics, ROC/PR curves, Confusion matrices
- **Balancing** — SMOTE (imbalanced-learn), Class weighting

---

## 📁 Repository Structure

```
├── Applied_AI_Project_final.ipynb   # Full implementation notebook
├── README.md                        # Project documentation
└── data/
    └── all_kindle_review.csv        # Amazon Kindle Reviews dataset
```

---

## 💡 Why This Matters

Sentiment analysis powers product recommendation engines, brand monitoring tools, and customer feedback systems across e-commerce, healthcare, and finance. This project demonstrates that **model complexity is not always the answer** — understanding your data and selecting the right baseline is just as important as architectural choices.

---

## 👤 Author

**Muhammad Salahuddin**  
MSc Artificial Intelligence & Data Science — University of Hull  
[GitHub](https://github.com/msnoorani) | [LinkedIn](https://linkedin.com/in/msnoorani)
