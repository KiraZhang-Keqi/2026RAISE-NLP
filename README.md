# RAISE-26 Multi-Label NLP Pipeline (DistilBERT + Baseline)

**Task:** Multi-label text classification on AI-related news headlines (Dataset A) + applying the trained model to LLM outputs (Dataset C).  
**Approach:** TF-IDF + One-vs-Rest Logistic Regression baseline, and DistilBERT fine-tuning with BCEWithLogitsLoss.  
**Key add-ons:** label cleaning + rare-label handling, multilabel-stratified split, thresholding, interpretability & error analysis, topic modeling (NMF), LLM framing comparison.

> This repository is a cleaned portfolio version of a school competition project.

---

## Project Highlights
- **End-to-end ML workflow:** data loading → preprocessing → encoding & stratified split → baseline → transformer → evaluation → interpretability → topic modeling → LLM output analysis.
- **Label quality handling:** semicolon-separated multi-label parsing + rare-label merge (`RARE_THRESHOLD=[TODO]`).
- **Reproducibility:** fixed seed, clear configs, scripts for training & evaluation.

---

## Dataset
### Dataset A (News Headlines)
- **Text column:** `title`
- **Label column:** `classes_str` (semicolon-separated multi-labels)
- **Note:** Dataset is NOT included in this repo (competition policy).  
  Place the file at: `data/dataset_A_news_full_10500.csv`

### Dataset C (LLM Outputs)
- File name variants supported (see `src/data_loading.py`).
- Place the file at: `data/Dataset_C_prompts___queries.csv`

---

## Methods
### 1) Preprocessing
- Text cleaning: lowercase, remove URLs/HTML, normalize whitespace.
- Label parsing: `classes_str` → `labels_list`.
- Rare labels: merge labels with frequency < `RARE_THRESHOLD` into `OTHER_RARE` (optional).

### 2) Baseline: TF-IDF + Logistic Regression (OvR)
- Text features: TF-IDF (ngram_range=[TODO])
- Optional metadata features: `source`, `day_of_week`, `month` + numerical features if available.
- Classifier: One-vs-Rest Logistic Regression (`class_weight="balanced"`)

### 3) Transformer: DistilBERT Multi-Label Classifier
- Model: DistilBERT encoder + linear head
- Loss: `BCEWithLogitsLoss`
- Training: AdamW + scheduler + gradient clipping
- Thresholding: default 0.5 or tuned thresholding (if enabled)

### 4) Evaluation
- Metrics: Micro-F1, Macro-F1, Weighted-F1, Samples-F1
- Per-label metrics + selected-label confusion matrices

### 5) Interpretability & Analysis
- Baseline: high-weight TF-IDF terms per label
- Error analysis: high-confidence false positives / false negatives
- Topic modeling: NMF over TF-IDF for latent themes
- Dataset C: compare label/cluster/topic distributions across LLMs & prompt types

---

## Results (Fill In)
| Model | Micro-F1 | Macro-F1 | Weighted-F1 | Samples-F1 |
|------|----------:|---------:|------------:|-----------:|
| TF-IDF + LogReg | [TODO] | [TODO] | [TODO] | [TODO] |
| DistilBERT | [TODO] | [TODO] | [TODO] | [TODO] |

Key findings:
- [TODO 1–3 bullets, e.g., DistilBERT improves semantic labels; baseline is more interpretable; rare labels remain challenging.]

---

## Repo Structure
```text
raise26-nlp-multilabel/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_final_pipeline.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocess.py
│   ├── split.py
│   ├── baseline_tfidf_lr.py
│   ├── bert_trainer.py
│   ├── metrics.py
│   ├── analysis_error.py
│   └── topic_nmf.py
├── scripts/
│   ├── train_baseline.py
│   ├── train_bert.py
│   ├── eval.py
│   └── predict_dataset_c.py
└── assets/
    ├── label_distribution.png
    ├── model_comparison.png
    └── nmf_topics.png
