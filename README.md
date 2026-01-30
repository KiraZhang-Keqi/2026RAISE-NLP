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

| Model | Micro-F1 | Macro-F1 |
|-------|----------|----------|
| **TF-IDF + Logistic Regression** | **0.9430** | **0.9331** |
| DistilBERT (Fine-tuned) | 0.9214 | 0.8862 |

> The traditional baseline outperformed the transformer model by 2.16pp (Micro-F1), demonstrating that sparse features with linear classifiers can be highly effective for short-text classification.

### Behavioral Categories

The 12-class taxonomy covers:

| Economic | Cognitive | Social |
|----------|-----------|--------|
| Work, Jobs & Economy | Cognitive & Decision-Making | Social Interaction & Relationships |
| Learning, Knowledge & Education | Creativity, Expression & Identity | Human Roles |
| Technology & Interaction | Emotion, Motivation & Well-being | Society, Ethics & Culture |
| Health, Safety & Risk | Sentiment (Positive/Negative) | Routine, Lifestyle & Behavior |

### LLM Comparison Insights

- **Llama**: Broadest behavioral coverage (7/12 categories), excels in lifestyle and emotional content
- **Qwen**: Specialized in Cognitive & Decision-Making (highest probability: 0.493)
- **Mistral**: Balanced approach, leads in educational and health topics
- **Key Finding**: All three LLMs show ~55-60% cluster overlap, suggesting convergent narrative framing

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

### Environment
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
| `RAISE-26 AI Behavioral Impact Analysis A Multi-Label NLP Text Classification Pipeline.ipynb` | Complete analysis pipeline (11 stages) |
| `RAISE-26 Presentation.pptx` | Project presentation slides |

## Acknowledgments

Built for **RAISE 2026** Research Competition.
