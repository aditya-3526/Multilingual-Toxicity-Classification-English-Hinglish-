# Multilingual Toxicity Classification

> A comparative study of **mBERT**, **XLM-RoBERTa**, and **Qwen2.5-0.5B + LoRA** on a unified English + Hinglish hate-speech corpus, with gradient-attribution explainability for code-mixed input.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-FFD21F)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue)](#)
[![Kaggle](https://img.shields.io/badge/Kaggle-T4%20×%202-20BEFF?logo=kaggle&logoColor=white)](#)

---

## TL;DR

Three transformer architectures from three different generations, fine-tuned on a 354,895-example merged corpus spanning English and Hinglish, evaluated head-to-head on six toxicity labels with full per-label and threshold-calibration analysis:

- **mBERT** (2019, encoder-only, 178 M params, full fine-tune)
- **XLM-RoBERTa** (2020, encoder-only, 278 M params, full fine-tune)
- **Qwen2.5-0.5B + LoRA** (2024, decoder-only LLM, 1.1 M trainable / 0.22 % of total)

**Headline result:** No single model wins. Qwen2.5 + LoRA achieves the best **F1 Micro (0.7362)** through higher recall on common labels; XLM-RoBERTa achieves the best **F1 Macro (0.6525)**, holding up better on the long tail (`threat`, `severe_toxic`). All three models score **ROC-AUC > 0.97** on rare labels — meaning the rare-class F1 deficit is a **threshold-calibration problem, not a representational one.**

---

## Why this project

Automated content moderation is a solved problem on English. It is not solved on **Hinglish** — the code-mixed Hindi-English register that dominates Indian-platform discourse. A user writing *"tu bahut stupid hai"* mixes Hindi grammar with English vocabulary and Roman transliteration, and a monolingual English classifier sees mostly noise.

We wanted to answer two questions head-on:

1. **Encoders vs decoder-LLMs:** does a 2024-era LLM with parameter-efficient adaptation actually beat a fully-fine-tuned encoder on this task, or is the LLM advantage overhyped at small scales?
2. **Aggregate vs per-label:** real deployments need uniform fidelity across the long tail of identity-targeted hate categories. Which model holds up when we stop hiding behind aggregate F1?

The findings turn out to be more interesting than "X beats Y." The right answer depends on which deployment constraint you care about — and the explainability layer makes that choice debuggable.

---

## Repository structure

```
.
├── bert-base-multilingual.ipynb      # mBERT training, evaluation, plots
├── xlm-r-merged.ipynb                # XLM-RoBERTa training, evaluation, plots
├── qwen2-5-merged.ipynb              # Qwen2.5-0.5B + LoRA training, evaluation, plots
├── cross-model-comparison.ipynb      # Aggregated bar charts, heatmap, radar plot
├── Toxicity_Classification_Report.pdf # Full 34-page project report
└── README.md
```

Each model notebook is self-contained: data loading → label harmonization → preprocessing → tokenization → fine-tuning → evaluation → per-label plots → confusion matrices → ROC / PR curves → gradient-attribution sample (XLM-RoBERTa). The cross-model notebook stitches the three runs together for the comparison figures.

---

## Method at a glance

### Data

Three datasets harmonized into a shared six-label binary schema (`toxic`, `obscene`, `insult`, `identity_hate`, `threat`, `severe_toxic`):

| Source | Language | Samples | Native labels | Mapping |
|---|---|---|---|---|
| Jigsaw Toxic Comment Challenge | English | 159,571 | Native 6-label | identity (no remap) |
| HASOC 2019–2021 | English + Hindi | ~16,000 | HOF / NOT (binary) | HOF → `[toxic=1, insult=1, others=0]` |
| MMHS150K | English (tweets) | 149,823 | 0=NotHate, 1–5=hate sub-types | any non-zero → `[toxic=1, insult=1, identity_hate=1, others=0]` |
| **Merged corpus** | **EN + HI** | **~354,895** | Six binary labels | concat + shuffle (seed=42) |

Note that `obscene`, `threat`, and `severe_toxic` are effectively **single-source labels** carried entirely by Jigsaw (HASOC and MMHS contribute zero positives for these three), which becomes the central interpretive frame for the rare-class results.

### Architectures

| Model | Type | Params (total) | Params (trainable) | Fine-tune strategy |
|---|---|---|---|---|
| mBERT (`bert-base-multilingual-cased`) | Encoder | 178 M | 178 M | Full |
| XLM-RoBERTa (`xlm-roberta-base`) | Encoder | 278 M | 278 M | Full |
| Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`) | Decoder LLM | 494 M | **1.1 M (0.22 %)** | LoRA r=16, α=32, target=`q_proj, v_proj` |

All three use Hugging Face's `AutoModelForSequenceClassification` with a 6-way multi-label classification head and `BCEWithLogitsLoss` (`problem_type="multi_label_classification"`). Optimizer: AdamW, weight decay 0.01, 10 % linear warmup, 3 epochs each.

### Hardware

Kaggle's free-tier dual Tesla T4 (16 GB VRAM each, 30 GB RAM). Wall-clock training time: mBERT ~2.5 h, XLM-RoBERTa ~3 h, Qwen2.5 + LoRA ~6.5 h.

---

## Results

### Aggregate performance

| Metric | mBERT | XLM-RoBERTa | **Qwen2.5 + LoRA** |
|---|---|---|---|
| F1 Micro | 0.7182 | 0.7270 | **0.7362** |
| F1 Macro | 0.6345 | **0.6525** | 0.6064 |
| F1 Weighted | 0.7179 | 0.7266 | **0.7353** |
| Precision (Micro) | 0.7374 | **0.7513** | 0.7353 |
| Recall (Micro) | 0.7000 | 0.7042 | **0.7371** |

### Per-label F1 — where it gets interesting

| Label | Support | mBERT | **XLM-R** | Qwen2.5 |
|---|---|---|---|---|
| toxic | 12,889 | 0.7416 | 0.7525 | **0.7593** |
| obscene | 850 | 0.7480 | **0.7672** | 0.7544 |
| insult | 12,175 | 0.7325 | 0.7395 | **0.7504** |
| identity_hate | 9,344 | 0.6702 | 0.6763 | **0.6907** |
| **threat** | **54** | 0.4600 | **0.4854** | 0.3243 |
| **severe_toxic** | **183** | 0.4545 | **0.4940** | 0.3594 |

On common labels, all three cluster within 1–2 points. On the two rarest labels, **Qwen2.5 collapses by ~15 points** while XLM-RoBERTa holds up — a direct consequence of LoRA's tiny adapter capacity being unable to learn the highly specialized boundaries needed for sparse classes.

### The threshold-calibration insight

ROC-AUC above 0.97 on `threat` and `severe_toxic` for all three models means they **do** rank toxic comments above non-toxic ones with very high fidelity. They just can't commit to a positive prediction at the canonical 0.5 threshold because the dominant negative class pushes positive-class probabilities downward.

| Label | mBERT AUC | XLM-R AUC | Qwen2.5 AUC |
|---|---|---|---|
| toxic | 0.9064 | 0.9139 | 0.9140 |
| obscene | 0.9884 | 0.9887 | 0.9900 |
| threat | 0.9722 | 0.9896 | 0.9825 |
| severe_toxic | 0.9926 | 0.9944 | 0.9944 |

Per-label threshold tuning on a held-out calibration split could plausibly close 5–10 pp of the rare-class F1 gap **without any change to the underlying models** — a deployment-relevant takeaway.

---

## Explainability — gradient attribution on Hinglish

A gradient-based attribution module sits on top of fine-tuned XLM-RoBERTa. For each input, it computes ∂ logit_toxic / ∂ embeddings, takes the absolute value, sums over the embedding dimension, and aligns scores back to surface tokens via the tokenizer's offset mapping.

Three patterns from running it on illustrative inputs:

| Sentence | High-attribution tokens |
|---|---|
| `you are a disgusting idiot` (toxic) | **idiot=0.91**, **disgusting=0.81** |
| `tu bahut stupid hai` (Hinglish toxic) | **bahut=1.00**, **stupid=0.64** |
| `you are a very kind person` (benign) | flat 0.65–1.00, no clear driver |
| `bhai tu mast kaam kar raha hai` (Hinglish benign) | flat 0.65–1.00, no clear driver |

The Hinglish case is the one that matters: the model correctly elevates the **Hindi intensifier** *bahut* ("very") alongside the English-origin toxic adjective *stupid*. That's evidence the SentencePiece vocabulary is encoding code-mixed semantics rather than treating Hindi tokens as out-of-vocabulary noise — which is the whole reason XLM-RoBERTa was the right backbone for this task.

---

## How to reproduce

These were authored as Kaggle notebooks against `/kaggle/input/` paths. To run locally:

```bash
git clone https://github.com/aditya-3526/Multilingual-Toxicity-Classification-English-Hinglish-.git
cd Multilingual-Toxicity-Classification-English-Hinglish-
pip install transformers datasets peft scikit-learn pandas numpy matplotlib seaborn torch
```

Datasets: download Jigsaw Toxic Comment Classification (Kaggle), HASOC 2019–2021, and MMHS150K from their respective sources. Update the dataset paths in cell 3 of each notebook (the `/kaggle/input/...` strings).

Run order:

1. `bert-base-multilingual.ipynb` — trains mBERT, dumps `final_metrics.csv` + plots
2. `xlm-r-merged.ipynb` — trains XLM-RoBERTa, dumps `final_metrics.csv` + plots
3. `qwen2-5-merged.ipynb` — trains Qwen2.5 + LoRA, dumps `final_metrics.csv` + plots
4. `cross-model-comparison.ipynb` — reads the three `final_metrics.csv` files and produces the aggregate comparison figures

Each notebook auto-saves visualizations as 300-DPI PNGs and writes a `final_metrics.csv` for downstream aggregation.

---

## Limitations

- **Single-source rare labels.** `obscene`, `threat`, and `severe_toxic` are carried entirely by Jigsaw, which limits cross-dataset generalization claims for these labels.
- **Label noise.** MMHS150K annotator disagreement is documented to be substantial; some reported errors are label noise rather than model failure.
- **Pre-training mismatch.** None of the three backbones was pre-trained on Indian-platform Hinglish specifically; this likely caps the upside on the most code-mixed validation samples.
- **Fixed 0.5 threshold.** All metrics are reported at the canonical sigmoid cutoff to ensure direct comparability. Per-label calibration is the obvious next step.
- **Compute-bounded.** 3 epochs and Kaggle T4 hardware. mBERT in particular shows continued F1 Macro improvement at epoch 3 (0.501 → 0.569 → 0.618), suggesting more epochs would help.

---

## What's in the report

The full project report (`Toxicity_Classification_Report.pdf`) covers everything in this README plus literature review, formal problem definition, derivations, training-dynamics figures, confusion matrices for all three models, error analysis (sarcasm, identity-mention false positives, veiled threats, severity downgrade), and complete reference list.

---

## Team

This project was built collaboratively as part of the Deep Learning course at Thapar Institute of Engineering and Technology.

| Member | Roll Number |
|---|---|
| Aditya Aryan | 102303526 |
| Mansehaj Preet Singh | 102303544 |
| Tanisha Dua | 102303545 |
| Ritisha Sidana | 102303552 |

Supervisor: **Kanupriya Mam**, Department of Computer Science and Engineering, TIET.

---

## Citation

If this work is useful to you, please cite the report:

```bibtex
@techreport{aryan2024multilingualtoxicity,
  title       = {Multilingual Toxicity Classification: A Comparative Study of mBERT, XLM-RoBERTa, and Qwen2.5},
  author      = {Aryan, Aditya and Singh, Mansehaj Preet and Dua, Tanisha and Sidana, Ritisha},
  institution = {Thapar Institute of Engineering and Technology},
  year        = {2024}
}
```

---

## References

Key works that informed the architecture and evaluation choices:

1. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
2. Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020. *(XLM-RoBERTa)*
3. Mandl et al. *Overview of the HASOC Track at FIRE 2019.* FIRE 2019.
4. Wulczyn et al. *Ex Machina: Personal Attacks Seen at Scale.* WWW 2017. *(Jigsaw schema)*
5. Gomez et al. *Exploring Hate Speech Detection in Multimodal Publications.* WACV 2020. *(MMHS150K)*
6. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
7. Yang et al. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024.
8. Sundararajan et al. *Axiomatic Attribution for Deep Networks.* ICML 2017. *(gradient attribution)*

Full reference list in Section 7 of the report.
