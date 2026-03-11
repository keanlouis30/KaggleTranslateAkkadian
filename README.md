# KaggleTranslateAkkadian
This repository contains a neural machine translation (NMT) pipeline designed to translate Akkadian transliterations into English. The project leverages the Helsinki-NLP/opus-mt-mul-en model, fine-tuned specifically for the semantic and grammatical nuances of the Akkadian language.

Project Overview
The objective of this project is to provide a computational tool for Assyriology by automating the translation of ancient cuneiform transliterations. The pipeline covers data preprocessing, sub-word tokenization, model fine-tuning, and BLEU score evaluation.

Technical Stack
Python Version: 3.12

Core Libraries: Transformers, PyTorch, Datasets, Evaluate, Scikit-learn, Pandas

Hardware Acceleration: Optimized for NVIDIA RTX 3060 using CUDA 12.6 and Mixed Precision (FP16)

Model Architecture: MarianMT (Seq2Seq Transformer)

Installation and Setup
1. Environment Configuration
It is recommended to use a virtual environment to manage dependencies and avoid system conflicts.

Bash
python -m venv .mvenv
.mvenv\Scripts\activate
2. Dependency Installation
Install the necessary packages via pip.

Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate evaluate scikit-learn pandas sacremoses sacrebleu
3. Dataset Placement
Ensure the Kaggle competition dataset is placed in the project root directory under the following folder structure:
deep-past-initiative-machine-translation/train.csv
deep-past-initiative-machine-translation/test.csv

Usage
Training the Model
To initiate the fine-tuning process, run the main training script. The script handles data cleaning, tokenization, and saves the best-performing model weights.

Bash
python train.py
Inference and Submission
Once training is complete, the script automatically performs inference on the test set and generates a submission.csv file formatted for Kaggle.

Model Performance
Training Time: Approximately 32 minutes on an RTX 3060.

Evaluation Metric: SacreBLEU.

Optimization: Utilizes Early Stopping and Gradient Accumulation to maintain stability and prevent over-fitting.

License
This project is intended for research and educational purposes in the field of computational linguistics and ancient language studies.