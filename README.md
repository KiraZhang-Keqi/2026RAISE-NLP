# RAISE-26 AI Behavioral Impact Analysis: A Multi-Label NLP Text Classification Pipeline
An end-to-end NLP pipeline for classifying news headlines into 12 behavioral impact categories, with comparative analysis of LLM-generated content and financial market correlations.

## Project Overview

This project analyzes **10,500+ news headlines** to understand how media frames AI's behavioral impact on society. The pipeline includes:

- **Multi-label text classification** with 12 behavioral categories
- **Model comparison**: TF-IDF + Logistic Regression vs. Fine-tuned DistilBERT
- **LLM narrative analysis**: Comparing outputs from Llama, Mistral, and Qwen
- **Topic modeling** with NMF for thematic discovery
- **Quantitative finance extension**: Correlating media sentiment with market movements

## Key Results

### Classification Performance

| Model | Micro-F1 | Macro-F1 | Weighted-F1 | Samples-F1 |
|-------|----------|----------|-------------|------------|
| **TF-IDF + Logistic Regression** | **0.9430** | **0.9331** | **0.9425** | **0.9216** |
| DistilBERT (Fine-tuned) | 0.9214 | 0.8862 | 0.9149 | 0.9115 |

> The traditional baseline outperformed the transformer model by 2.16pp (Micro-F1), demonstrating that sparse features with linear classifiers can be highly effective for short-text classification.

### Per-Label Performance (Top & Bottom)

| Category | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Work, Jobs & Economy | 1.000 | 0.965 | **0.982** |
| Learning, Knowledge & Education | 0.984 | 0.959 | 0.971 |
| Technology & Interaction | 0.990 | 0.936 | 0.962 |
| ... | ... | ... | ... |
| Emotion, Motivation & Well-being | 1.000 | 0.800 | 0.889 |
| Cognitive & Decision-Making | 0.807 | 0.893 | 0.848 |

> Categories with clear lexical markers (e.g., "jobs", "automation") achieve near-perfect precision, while abstract concepts (e.g., cognitive, emotional) show semantic overlap with adjacent categories.

### Behavioral Categories

The 12-class taxonomy covers:

| Economic | Cognitive | Social |
|----------|-----------|--------|
| Work, Jobs & Economy | Cognitive & Decision-Making | Social Interaction & Relationships |
| Learning, Knowledge & Education | Creativity, Expression & Identity | Human Roles |
| Technology & Interaction | Emotion, Motivation & Well-being | Society, Ethics & Culture |
| Health, Safety & Risk | Sentiment (Positive/Negative) | Routine, Lifestyle & Behavior |

### Topic Modeling (NMF)

Discovered 10 latent topics from the corpus:

| Topic | Top Keywords |
|-------|--------------|
| Topic 0 | artificial intelligence, stock, prediction, technology |
| Topic 1 | ai innovation, future, data governance, global tech |
| Topic 2 | healthcare, medical diagnosis, patient care |
| Topic 3 | jobs, workers, automation, employment |
| Topic 4 | education, students, learning, classroom |

### LLM Comparison Insights

| LLM | Dominant Categories | Topic Diversity (Entropy) |
|-----|---------------------|---------------------------|
| **Llama** | 7/12 categories (broadest) | High |
| **Mistral** | 3/12 categories | Highest |
| **Qwen** | 2/12 categories (focused) | Lower |

- **Llama**: Excels in lifestyle, emotional, and ethical content
- **Qwen**: Specialized in Cognitive & Decision-Making (probability: 0.493)
- **Mistral**: Leads in educational, health, and social interaction topics
- **Key Finding**: All three LLMs show ~55-60% cluster overlap, suggesting convergent narrative framing despite different architectures

### Statistical Validation

- **Chi-Square Test**: Significant differences in label distributions across LLMs (p < 0.05)
- **Label Distribution**: Maintained consistent across train/val/test splits via stratified sampling
- **Reproducibility**: All experiments seeded (SEED=42) with deterministic CUDA operations

## Pipeline Architecture

```
Data Loading → Preprocessing → Multi-Label Encoding → Train/Val/Test Split
                                                              ↓
                              ┌─────────────────────────────────────────────┐
                              │                                             │
                              ▼                                             ▼
                    TF-IDF + LogReg                              DistilBERT Fine-tuning
                    (Baseline Model)                             (Deep Learning Model)
                              │                                             │
                              └──────────────┬──────────────────────────────┘
                                             ▼
                                   Model Evaluation & Comparison
                                             ↓
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
             Topic Modeling           LLM Output Analysis      Quant Finance Extension
               (NMF)                (Llama/Mistral/Qwen)       (Market Correlation)
```

## Tech Stack

**Core ML/NLP:**
- scikit-learn (TF-IDF, Logistic Regression, NMF)
- PyTorch + HuggingFace Transformers (DistilBERT)
- NLTK (Text preprocessing)

**Analysis & Visualization:**
- pandas, numpy
- matplotlib, seaborn
- scipy (Statistical testing)

**Finance Extension:**
- yfinance, arch (GARCH modeling)

## Environment

#### Run on Google Colab (Recommended)
```bash
# If running in Google Colab, uncomment the following lines.
# !pip -q install -U nltk iterative-stratification
# !pip -q install -U transformers datasets accelerate

# Local Environment Setup
# pip install pandas numpy matplotlib seaborn nltk scikit-learn transformers torch accelerate iterative-stratification
import os
import re
import json
import warnings
from typing import List, Dict, Optional, Tuple

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP utilities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score
)
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# Statistical testing (for quantitative analysis)
from scipy import stats
from scipy.stats import chi2_contingency, entropy

# Multi-label stratified sampling
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    ITERSTRAT_AVAILABLE = True
except ImportError:
    ITERSTRAT_AVAILABLE = False
    print("Warning: iterative-stratification is not installed. Falling back to random splitting.")

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Transformers
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Global settings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_colwidth", 100)

print("✓ All libraries imported successfully.")

#Quant Analysis Dependencies
!pip install -q yfinance arch

import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from arch import arch_model
```

## Files

| File | Description |
|------|-------------|
| `RAISE-26_Competition_Guidelines.pdf` | Project Guidelines |
| `RAISE-26 AI Behavioral Impact Analysis A Multi-Label NLP Text Classification Pipeline.ipynb` | Complete analysis pipeline (11 stages) |
| `RAISE-26 Presentation.pptx` | Project presentation slides |

## Acknowledgments

Built for **RAISE 2026** Research Competition.
