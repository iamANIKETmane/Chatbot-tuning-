# Chatbot Model - HPML Homework 3

## Overview
This project involves building a chatbot using **movie dialogues** from the Convokit dataset. The dataset contains **220,579 conversational exchanges** between **10,292 pairs of movie characters**, extracted from **617 movies**. The chatbot is developed using **PyTorch** and leverages deep learning techniques to generate responses.

## Dataset
- **Source:** Convokit Movie Corpus
- **Total Conversations:** 220,579
- **Characters:** 9,035
- **Total Utterances:** 304,713
- **Format:** JSONL files (utterances.jsonl)

## Model Architecture
The chatbot follows a **sequence-to-sequence (seq2seq) architecture** with the following components:
- **Preprocessing:** Data cleaning, tokenization, and normalization.
- **Word Embeddings:** Word2Vec, GloVe, or learned embeddings.
- **Model:** RNN, LSTM, or Transformer-based encoder-decoder.
- **Training:** Optimization using loss functions (e.g., Cross-Entropy Loss).
- **Inference:** Generating responses based on trained weights.
- **Evaluation Metrics:** BLEU Score, Perplexity, etc.

## Workflow
1. **Dataset Loading** - Movie dialogues are extracted from Convokit.
2. **Preprocessing** - Text is cleaned, tokenized, and normalized.
3. **Embedding Layer** - Word embeddings (e.g., Word2Vec, GloVe) are applied.
4. **Seq2Seq Model Training** - The chatbot is trained using LSTM/Transformer models.
5. **Inference** - Given an input, the model generates a conversational response.
6. **Evaluation** - Performance is measured using BLEU Score, Perplexity, etc.

## Flowchart
The chatbot model follows this pipeline:

![Chatbot Model Pipeline](./flowchart.png)

## References
1. **Cornell Movie Dialogs Corpus** - Danescu-Niculescu-Mizil et al. (2011)
   - [Paper](https://arxiv.org/abs/1106.3077)
   - [Dataset](https://zissou.infosci.cornell.edu/convokit/datasets.html)
2. **Sequence-to-Sequence Learning with Neural Networks** - Sutskever et al. (2014)
   - [Paper](https://arxiv.org/abs/1409.3215)
3. **Attention Is All You Need** - Vaswani et al. (2017)
   - [Paper](https://arxiv.org/abs/1706.03762)

## Installation
To set up the environment, run:
```bash
pip install torch numpy nltk
```

## Running the Notebook
1. Load the dataset:
   ```bash
   !wget -O data/movie-corpus.zip https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip
   !unzip -o data/movie-corpus.zip -d data
   ```
2. Train the chatbot model using the notebook.
3. Generate responses by providing input queries.

---
This project demonstrates **Natural Language Processing (NLP) for conversational AI**, implementing **deep learning techniques** to create a functional chatbot.

