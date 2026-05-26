# DAα: Inference-Time Semantic Modulation for Audience-Aware LLM Generation

Anonymous repository for EMNLP 2026 submission.

---

## Overview

This repository contains the implementation, evaluation scripts, and experimental results for **DAα**, a lightweight inference-time framework for controllable audience adaptation in Large Language Models (LLMs).

DAα performs:

- semantic filtering  
- embedding-space interpolation  
- inference-time semantic modulation  
- controllable audience adaptation  

without requiring:

- fine-tuning  
- adapters  
- retraining  
- RLHF  
- architecture modifications  

The framework enables continuous adaptation between novice-oriented and expert-oriented responses using a single interpolation parameter:

\[
\alpha \in [0,1 - 0.9]
\]

where:

- lower values produce simpler explanations  
- higher values produce more domain-specific and technically detailed outputs  

The primary evaluation domain is climate-law/legal-compliance communication, with additional qualitative robustness validation in programming-language explanations.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── RQ1/
│   ├── RQ2/
├── results/
│   ├── rq1/
│   ├── rq2/
│   ├── programming/

## Requirements

The project was developed using Python 3.10.8.

Main dependencies include:

- transformers  
- sentence-transformers  
- torch  
- numpy  
- pandas  
- scikit-learn  
- bert-score  
- rouge-score  
- nltk  

Install dependencies with:

```bash
pip install -r requirements.txt
```

Method Summary

DAα combines:

## Domain semantic filtering
Topic-level semantic constraints
Embedding-space interpolation
Inference-time audience modulation

The framework operates by constructing steering representations from domain/topic embeddings and modulating generation through interpolation controlled by α.

No parameter updates or model retraining are required.

## Programming-Domain Validation

Supplementary qualitative robustness validation using programming-language questions.

The programming domain is included as a secondary validation setting to evaluate whether DAα preserves controllable semantic modulation outside the primary legal-compliance domain.

## Reproducibility

All experiments were executed using lightweight open-weight LLMs under identical evaluation settings.

## The repository includes:

evaluation scripts
prompt-only baselines
steering configurations

## Ethical Considerations

This repository is released for research purposes only.

DAα modulates explanation complexity and lexical specificity but does not guarantee factual correctness. Outputs should not be used in high-stakes legal, medical, or safety-critical environments without human verification.

No personally identifiable information (PII) is included in the released artifacts.

## Citation

If this work is accepted, citation information will be updated in the final repository version.

## Anonymous Submission Notice

This repository is intentionally anonymized for double-blind review. Identifying metadata, affiliations, and author information have been removed.