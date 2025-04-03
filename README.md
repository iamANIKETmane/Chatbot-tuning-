# Chatbot Project

## Overview
This project involves training a chatbot using a sequence-to-sequence model based on the Cornell Movie Dialogs Corpus. The chatbot is implemented using PyTorch and follows concepts from the PyTorch Chatbot Tutorial.

## Approach
1. **Dataset Preparation**
   - Loaded and preprocessed the Cornell Movie-Dialogs Corpus.
   - Extracted conversational pairs for training.

2. **Model Development**
   - Implemented a sequence-to-sequence (seq2seq) model with Luong attention.
   - Trained an encoder-decoder architecture using mini-batches.
   - Implemented greedy-search decoding for response generation.

3. **Hyperparameter Tuning with Weights & Biases (W&B)**
   - Integrated W&B for logging and tracking experiments.
   - Configured a hyperparameter sweep using Random Search for:
     - Learning rate: [0.0001, 0.00025, 0.0005, 0.001]
     - Optimizer: [adam, sgd]
     - Clip: [0, 25, 50, 100]
     - Teacher forcing ratio: [0, 0.5, 1.0]
     - Decoder learning ratio: [1.0, 3.0, 5.0, 10.0]
   - Conducted hyperparameter tuning on GPU-enabled Colab.
   - Analyzed the best hyperparameter set based on minimum loss.

4. **Performance Profiling**
   - Used PyTorch Profiler to measure execution time and memory consumption.
   - Analyzed CUDA kernel performance in Chrome Trace Viewer.

5. **Model Evaluation and Saving**
   - Evaluated the trained chatbot model.
   - Saved the best-performing model based on loss.

---

1. **Understanding TorchScript**
   - Compared tracing and scripting methods:
     - Tracing: Runs a sample input through the model and records operations.
     - Scripting: Converts model code into a TorchScript program.

2. **Modifying Model for Scripting**
   - Replaced incompatible Python constructs with TorchScript-friendly ones.
   - Used `torch.jit.script` instead of tracing to handle dynamic loops and conditionals.

3. **TorchScript Model Conversion**
   - Converted the trained chatbot model to TorchScript.
   - Printed the TorchScript computation graph.
   - Evaluated TorchScript model performance.

4. **Latency Comparison**
   - Measured evaluation latency of both PyTorch and TorchScript models.
   - Compared CPU and GPU performance.
   - Created a latency table showing speedup improvements.

5. **Deployment for Non-Python Environments**
   - Saved and serialized the TorchScript model.
   - Demonstrated loading the model in C++ using LibTorch.

## Features
- Loads and preprocesses movie dialogue data.
- Implements a seq2seq chatbot with attention mechanism.
- Supports hyperparameter tuning with W&B.
- Profiles performance using PyTorch Profiler.
- Converts and optimizes the model for TorchScript deployment.

## Installation
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries (install using `requirements.txt` if available)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Chatbot.ipynb
   ```
2. Run the notebook cells in sequence to preprocess data and train the model.
3. Interact with the chatbot in the final section of the notebook.

## Dataset
The dataset used is the Cornell Movie-Dialogs Corpus. It is loaded and processed within the notebook.

## License
Specify the license under which the project is distributed (e.g., MIT License).

