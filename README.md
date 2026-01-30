# RAISE-26 AI Behavioral Impact Analysis: A Multi-Label NLP Text Classification Pipeline

An end-to-end NLP pipeline for classifying news headlines into 12 behavioral impact categories, with comparative analysis of LLM-generated content and financial market correlations.

> **Research Question:** How does media discourse frame AI's impact on human behavior?

---

## Project Overview

This project analyzes **10,500+ news headlines** to understand how media frames AI's behavioral impact on society. The pipeline includes:

- **Multi-label text classification** with 12 behavioral categories
- **Model comparison**: TF-IDF + Logistic Regression vs. Fine-tuned DistilBERT
- **LLM narrative analysis**: Comparing outputs from Llama, Mistral, and Qwen
- **Topic modeling** with NMF for thematic discovery
- **Quantitative finance extension**: Correlating media sentiment with market movements

---

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

---

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

---

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

---

## Quick Start

### Run on Google Colab (Recommended)

1. Open `2026RAISE_NLP.ipynb` in Google Colab
2. Upload the required datasets when prompted
3. Run all cells sequentially

### Local Environment

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn torch transformers accelerate iterative-stratification yfinance arch
```

---

## Files

| File | Description |
|------|-------------|
| `2026RAISE_NLP.ipynb` | Complete analysis pipeline (11 stages) |
| `Presentation.pptx` | Project presentation slides |

---

## Acknowledgments

Built for **RAISE 2026** Research Competition.

---

## Contact

Feel free to reach out for questions or collaboration opportunities.

- **Email**: [your.email@example.com]
- **LinkedIn**: [Your Profile]
