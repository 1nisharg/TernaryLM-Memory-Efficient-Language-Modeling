# TernaryLM  
**Native 1-Bit (Ternary) Language Modeling with Adaptive Layer-wise Scaling**

> Memory-efficient language modeling via native ternary quantization `{−1, 0, +1}` with stable training, competitive performance, and near-parity inference latency.

---

## Overview

**TernaryLM** is a 132M-parameter decoder-only Transformer trained **from scratch** using **1-bit ternary weights**, demonstrating that extreme quantization can be a *training paradigm*, not just a post-hoc compression trick.

Unlike post-training quantization (PTQ), TernaryLM **learns representations under quantization constraints** from the first optimization step, enabling the model to adapt internally to discrete weights.


---

## Why TernaryLM?

Large language models are constrained by:
- GPU memory
- inference cost
- deployment feasibility on edge / consumer devices

Most existing approaches:
- train in FP16/FP32  
- compress *after* training (INT8 / 4-bit PTQ)

**TernaryLM asks a different question**:

> *Can a language model be trained natively with 1-bit precision while preserving useful linguistic representations?*

This work shows the answer is **yes**, under the right optimization and architectural choices.

---

## Key Contributions

- **Native 1-bit training** using ternary weights `{−1, 0, +1}`
- **Adaptive per-layer scaling (α)** for stable optimization
- **Straight-Through Estimator (STE)** for gradient propagation
- **Stable convergence** across datasets with different entropy profiles
- **2.4× inference memory reduction** vs FP32 baselines
- **Near-parity per-token latency** without custom kernels
- **Layer-wise quantization analysis** revealing non-uniform sensitivity

---
## Training Setup:

- **Optimizer:** AdamW  
- **Learning rate:** Cosine schedule (peak 1e-3, 1k warmup)  
- **Batch:** 64 × 512 tokens (32k tokens / step)  
- **Epochs:** 15  
- **Hardware:** Dual NVIDIA T4 (16GB)  
- **Loss:** Autoregressive cross-entropy + label smoothing (0.1)


## Training Stability:
TernaryLM exhibits **smooth, monotonic loss decay** under 1-bit constraints.

Validated on:
- **TinyStories** (controlled, low entropy)
- **WritingPrompts** (creative, high entropy)

This demonstrates that instability is *not inherent* to extreme quantization — it is an optimization problem.

### Downstream Transfer (GLUE)

| Task | Metric | TernaryLM | BERT-Base |
|----|------|-----------|----------|
| MRPC | F1 | **82.47** | 84.98 |
| SST-2 | Acc | 88.92 | 92.43 |
| CoLA | MCC | 47.23 | 56.78 |

Quantized representations remain **semantically useful**, especially for similarity and classification tasks.

# Efficiency

### Inference Memory (GPU)

| Model | Memory (MB) |
|----|------------|
| BERT-Base FP32 | 1197 |
| BERT-Base INT8 | 612 |
| **TernaryLM (1-bit)** | **498** |

**2.4× reduction vs FP32**

---

### Per-Token Latency

| Model | ms/token |
|----|----------|
| FP32 | 9.52 |
| INT8 | 8.21 |
| **TernaryLM** | **9.41** |

Near-parity latency **without specialized ternary kernels**.


TernaryLM is **inspired by [BitNet](https://arxiv.org/abs/2310.11453)**, but differs in scope and intent:

| Aspect | BitNet | TernaryLM |
|------|-------|----------|
| Target scale | Billions+ | 132M |
| Hardware | Large clusters | Single GPU |
| Precision | 1 / 1.58-bit | **Pure ternary (1-bit)** |
| Focus | Scaling laws | **Practical efficiency & analysis** |

TernaryLM complements BitNet by showing what is feasible under **realistic resource constraints**.

