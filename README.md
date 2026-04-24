# Banking Intent Classification with DistilBERT

This project builds a transformer-based intent classification model for banking customer support queries. The goal is to classify incoming customer messages into predefined intent categories so that support teams can route requests more consistently and reduce manual triage effort.

The project uses a subset of the Banking77 dataset and fine-tunes DistilBERT for multiclass text classification.

## Project Overview

Customer support teams often receive large volumes of short, unstructured messages. Before a request can be resolved, it usually needs to be understood and routed to the right support category. Manual triage can be slow and inconsistent, especially when customer messages are written in different ways.

This project explores how a fine-tuned transformer model can support first-stage intent classification for banking-related queries.

## Objectives

- Prepare and filter banking customer support text data
- Fine-tune DistilBERT for intent classification
- Evaluate model performance using accuracy, weighted F1-score, confusion matrix, and class-level metrics
- Review prediction behavior on sample customer queries
- Build a simple Gradio demo for testing new customer queries

## Dataset

This project uses the Banking77 dataset, a public dataset of banking-related customer service queries labeled by intent.

For this prototype, the model is trained on 10 selected intent categories to keep the project focused and easier to evaluate.

## Tools and Libraries

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Gradio

## Methodology

1. Loaded and explored the Banking77 dataset
2. Selected a subset of 10 banking intent categories
3. Tokenized customer queries using a DistilBERT tokenizer
4. Fine-tuned DistilBERT using Hugging Face Trainer
5. Evaluated performance on the validation set
6. Reviewed model predictions and error patterns
7. Created a simple interactive demo using Gradio

## Model Performance

The fine-tuned DistilBERT model achieved strong validation performance in this 10-class prototype setting.

## Example Prediction

Sample query:

> My card was swallowed by the ATM and I need help.

Example top predicted intents:
1. declined_cash_withdrawal
2. wrong_amount_of_cash_received
3. balance_not_updated_after_cheque_or_cash_deposit

Because this prototype uses only 10 selected intent categories, some real-world queries may be mapped to the closest available intent rather than an exact category.

## Practical Relevance

A model like this could support customer support teams by helping with first-stage triage. It can suggest likely intent categories for incoming messages, which may help route requests more consistently. This project is a prototype and is not intended as a production banking system.

## Limitations

- The model uses only 10 selected Banking77 intent categories
- Some customer queries may not match the available category set exactly
- The dataset is public and may not fully represent real production support traffic
- Further testing would be needed before using this type of system in a real support workflow

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
