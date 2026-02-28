# 🏆 TOPSIS Analysis for Text Classification Models

> **Multi-Criteria Decision Making to Rank Pre-Trained NLP Models**

---

## 📋 Assignment Details

| Property | Details |
|---|---|
| **Roll Number** | 102317299 |
| **Task** | Text Classification |
| **Method** | TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) |

---

## 🧾 Overview

This project applies the **TOPSIS** multi-criteria decision-making method to evaluate and rank 10 pre-trained transformer models for text classification. TOPSIS identifies the optimal model by simultaneously considering multiple performance and efficiency criteria, ranking each alternative based on its geometric distance from ideal and negative-ideal solutions.

---

## 📁 Project Structure

```
assignment_3/
├── topsis.py                       # TOPSIS algorithm implementation
├── text_classification_topsis.py   # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── topsis_results.csv              # Detailed results table (generated)
├── topsis_scores.png               # TOPSIS scores visualization
├── radar_chart_top5.png            # Top 5 models comparison
├── performance_efficiency.png      # Performance vs efficiency analysis
├── criteria_weights.png            # Criteria weights visualization
└── ideal_distances.png             # Distance from ideal solutions
```

---

## 🤖 Models Evaluated

| # | Model | Description |
|---|---|---|
| 1 | **BERT-base** | Bidirectional Encoder Representations from Transformers (base) |
| 2 | **BERT-large** | BERT with larger architecture |
| 3 | **RoBERTa-base** | Robustly Optimized BERT Pretraining Approach (base) |
| 4 | **RoBERTa-large** | RoBERTa with larger architecture |
| 5 | **DistilBERT** | Distilled, faster and smaller version of BERT |
| 6 | **ALBERT-base** | A Lite BERT (base) |
| 7 | **ALBERT-xxlarge** | ALBERT with extra-large architecture |
| 8 | **ELECTRA-base** | Efficiently Learning an Encoder (base) |
| 9 | **DeBERTa-base** | Decoding-enhanced BERT (base) |
| 10 | **DeBERTa-large** | DeBERTa with larger architecture |

---

## ⚖️ Evaluation Criteria & Weights

| Criterion | Direction | Weight | Rationale |
|---|---|---|---|
| Accuracy (%) | MAX ↑ | **0.25** | Primary performance metric |
| F1 Score | MAX ↑ | **0.25** | Primary performance metric |
| Inference Speed (tokens/sec) | MAX ↑ | 0.15 | Deployment efficiency |
| Training Time (hours) | MIN ↓ | 0.15 | Fine-tuning cost |
| Model Size (MB) | MIN ↓ | 0.10 | Storage footprint |
| Memory Usage (GB) | MIN ↓ | 0.10 | Runtime resource cost |

> Accuracy and F1 Score carry the highest weights (25% each) as they are the primary indicators of model quality for text classification.

---

## 🔬 TOPSIS Methodology

TOPSIS ranks alternatives by evaluating their closeness to a **Positive Ideal Solution (PIS)** and distance from a **Negative Ideal Solution (NIS)**.

### Algorithm Steps

```
1. Normalize the decision matrix  →  eliminate scale differences across criteria
2. Apply criterion weights         →  reflect relative importance
3. Identify ideal solutions        →  PIS (best values), NIS (worst values)
4. Calculate Euclidean distances   →  distance from PIS and NIS for each model
5. Compute closeness coefficient   →  TOPSIS Score
6. Rank models                     →  higher score = better overall choice
```

### Scoring Formula

$$\text{Score} = \frac{D^-}{D^+ + D^-}$$

Where $D^+$ = distance from positive ideal and $D^-$ = distance from negative ideal. A score closer to **1.0** indicates a superior alternative.

---

## 📊 Results

### Top 3 Recommended Models

| Rank | Model | TOPSIS Score | Key Strengths |
|---|---|---|---|
| 🥇 1 | **DistilBERT** | **0.9536** | Best balance of speed, size & accuracy |
| 🥈 2 | **ELECTRA-base** | **0.7999** | Strong accuracy with moderate efficiency |
| 🥉 3 | **ALBERT-base** | **0.7937** | Smallest size with solid performance |

### Detailed Top 3 Breakdown

**1. DistilBERT** (Score: 0.9536)
- Accuracy: 91.2% | F1: 0.912
- Inference Speed: 680 tokens/sec
- Model Size: 260 MB | Memory: 1.2 GB

**2. ELECTRA-base** (Score: 0.7999)
- Accuracy: 93.5% | F1: 0.935
- Inference Speed: 520 tokens/sec
- Reasonable resource usage across all metrics

**3. ALBERT-base** (Score: 0.7937)
- Accuracy: 92.8% | F1: 0.928
- Model Size: 230 MB | Memory: 1.8 GB
- Most compact model with competitive accuracy

### Key Findings

- **Best for Production:** DistilBERT, ELECTRA-base, or ALBERT-base — excellent performance with manageable resource requirements
- **Best for Research / Max Accuracy:** DeBERTa-large (95.8%) or ALBERT-xxlarge (95.5%) — highest accuracy but resource-intensive
- **Core Trade-off:** Larger models score lower in TOPSIS despite higher raw accuracy, due to penalization from slow inference and high memory usage

---

## 📈 Visualizations Generated

| File | Description |
|---|---|
| `topsis_scores.png` | Bar chart of TOPSIS scores across all 10 models |
| `radar_chart_top5.png` | Radar chart comparing top 5 models on all criteria |
| `performance_efficiency.png` | Scatter plot of performance vs. efficiency trade-offs |
| `criteria_weights.png` | Bar chart of assigned criterion weights |
| `ideal_distances.png` | Distance from positive and negative ideal solutions |

---

## 🚀 Installation & Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the analysis**
```bash
python text_classification_topsis.py
```

This will perform the full TOPSIS analysis, print ranked results to console, save `topsis_results.csv`, and generate all visualization PNGs.

---

## 🧠 Implementation Notes

The TOPSIS algorithm was implemented from scratch in Python using the following approach:

- **Normalization:** Vector normalization to bring all criteria to a comparable scale
- **Weighting:** Element-wise multiplication of normalized matrix with criterion weights
- **Ideal Solutions:** Per-column max/min depending on benefit or cost criteria
- **Distance Metric:** Euclidean distance in n-dimensional criteria space
- **Ranking:** Closeness coefficient sorted in descending order

Data metrics were sourced from published benchmarks (GLUE, SuperGLUE), Hugging Face model cards, and research papers.

---

## 📌 Conclusion

TOPSIS provides a principled, transparent framework for comparing NLP models across competing objectives. This analysis shows that raw accuracy alone is not sufficient for model selection — efficiency, memory, and speed are equally important in real-world deployment. DistilBERT emerges as the optimal choice for most text classification scenarios, while larger models remain relevant for research contexts where accuracy is the sole priority.

---

## 📎 References

- Hwang, C.L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*
- Hugging Face Transformers — [huggingface.co](https://huggingface.co)
- GLUE / SuperGLUE Benchmarks — [gluebenchmark.com](https://gluebenchmark.com)
- Papers with Code — [paperswithcode.com](https://paperswithcode.com)

---

*Assignment 3 — Roll Number: 102317299 | Text Classification using TOPSIS*
