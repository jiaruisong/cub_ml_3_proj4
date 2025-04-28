## Problem & Data Description

The goal is to classify tweets as describing **real disasters** (1) or not (0). The training set contains **7,613 tweets** with 5 columns (*id, text, keyword, location, target*). The average cleaned tweet length is **15.2 words** (σ=6.0).

## Exploratory Data Analysis

See Figure 1 – 3 for length distribution, label balance, and missing‑value heatmap.

## Model Architecture

We use a **Bi‑LSTM** because … (captures forward & backward context, avoids word‑order loss). Input tweets are tokenised, padded to 50 tokens, and passed through a 100‑D GloVe embedding. A GlobalMaxPooling layer condenses sequence information, followed by a 64‑unit ReLU dense layer and a sigmoid output. Total parameters: ~2.1 M (100 K trainable).

## Results & Analysis

Validation F1 after tuning: **0.763** (Acc 0.813) after 4 epochs. Early stopping prevented over‑fitting (see Fig‑4 loss curves). Hyper‑parameter trials showed embedding dimensionality (50 vs 100) had modest impact, while bidirectionality consistently improved F1 by ~0.02.

## Conclusion & Future Work

The Bi‑LSTM surpasses the TF‑IDF + LogReg baseline by ~5 F1 points. Future work: • try RoBERTa‑base fine‑tune, • augment with keyword/location features, • ensemble multiple folds.

