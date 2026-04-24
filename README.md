# Banking Intent Classification with DistilBERT

This project builds a transformer-based intent classification system for banking customer support queries. The goal is to classify short customer messages into likely intent categories so they can be routed more consistently to the right support workflow.

The project uses a subset of the Banking77 dataset and fine-tunes DistilBERT for multiclass text classification. A Gradio interface is also included to demonstrate how the model can be used as a lightweight support-triage prototype.

---

## Project Overview

Customer support teams in financial services receive large volumes of short, high-variability text queries related to cards, cash withdrawals, transfers, balance updates, and other account issues. Manual triage can be slow and inconsistent, especially when messages are phrased differently but refer to similar underlying issues.

This project explores whether a transformer-based model can support first-stage intent detection for banking support queries.

---

## Problem Statement

Given a short customer support query, predict the most likely banking intent category from a selected set of intent classes.

This is framed as a **multiclass text classification** problem.

---

## Dataset

This project uses the **Banking77** dataset, a public dataset containing banking-related customer service queries labeled by intent.

For this prototype, the workflow focuses on a **10-intent subset** to keep the problem scoped, interpretable, and suitable for demonstration.

---

## Objectives

- Load and explore banking customer support queries
- Select a focused subset of banking intent categories
- Fine-tune DistilBERT for multiclass intent classification
- Evaluate the model using validation metrics and class-level analysis
- Review prediction behavior on realistic sample queries
- Demonstrate the workflow through a Gradio-based interface

---

## Tools and Libraries

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Gradio

---

## Methodology

1. Load the Banking77 dataset
2. Select 10 target intent categories
3. Remap labels for the selected subset
4. Tokenize customer queries using the DistilBERT tokenizer
5. Fine-tune DistilBERT using Hugging Face Trainer
6. Evaluate performance on a validation split
7. Review confusion matrix, classification report, and sample predictions
8. Launch a Gradio demo for interactive testing

---

## Model Output

For each input query, the model returns the **top predicted intent categories** with confidence scores.

Example query:

> My card was swallowed by the ATM and I need help.

Example model output may include categories such as:

- Declined Cash Withdrawal
- Wrong Amount Of Cash Received
- Balance Not Updated After Cheque Or Cash Deposit

Because this prototype uses a limited set of selected intents, some real-world queries may be mapped to the closest available category rather than an exact class.

---

## Gradio Demo

The project includes a Gradio-based demo interface for testing banking support queries interactively. Users can enter a customer query and view the top predicted intent categories returned by the model.

### Demo Screenshots

#### Gradio Home Interface
![Gradio Home Interface](screenshots/gradio_screenshot1.png)


![Example Query Prediction](screenshots/gradio_screenshot2.png)


![Another Query Prediction Example](screenshots/gradio_screenshot3.png)

---

## Practical Relevance

A system like this can support customer support operations by helping with first-stage triage. Instead of manually reading each incoming query from scratch, teams could use model predictions as an assistive signal for queue assignment or issue routing.

This project is intended as a **decision-support prototype**, not a production banking system.

---

## Limitations

- The model is trained on only a subset of Banking77 intent categories
- Some user queries may not map cleanly to the available classes
- Performance is based on a prototype setup and not a production evaluation environment
- Additional testing, monitoring, and human review would be needed for real-world deployment

---

## Repository Contents

- `banking_intent_classification.py` — main project script
- `requirements.txt` — Python dependencies
- `README.md` — project documentation

---

## Environment Setup

It is recommended to run this project in a fresh conda environment.

```bash
conda create -n banking-intent python=3.11 -y
conda activate banking-intent
pip install torch pandas matplotlib seaborn scikit-learn transformers datasets gradio accelerate pyarrow
