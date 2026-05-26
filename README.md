# 🏥 Medical Report Summariser

> AI-powered tool that converts complex clinical discharge reports into plain-English patient summaries using fine-tuned BART with LoRA.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/spaces/Nandini2217/medical-report-summariser)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Problem

1 in 3 patients cannot understand their hospital discharge papers. Complex medical jargon leads to missed follow-ups, medication errors, and preventable re-admissions.

## 💡 Solution

Fine-tuned BART-large with LoRA on medical text — produces plain-English summaries patients can actually understand in under 2 seconds.

## 📊 Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3392 |
| ROUGE-2 | 0.1310 |
| ROUGE-L | 0.2708 |
| Text Reduction | 61.5% |
| Readability Improvement | +16.2 pts (Flesch Reading Ease) |
| Trainable Parameters | 0.63% (LoRA) |

## 🚀 Live Demo

👉 [Try it on HuggingFace Spaces](https://huggingface.co/spaces/Nandini2217/medical-report-summariser)

## 📁 Project Structure

```
medical-report-summariser/
├── data/
│   ├── raw/              # Original datasets (gitignored)
│   ├── processed/        # Train/val/test splits (gitignored)
│   └── samples/          # Sample records for testing
├── models/
│   ├── checkpoints/      # Training checkpoints (gitignored)
│   └── final/            # Fine-tuned LoRA weights
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading from HuggingFace / CSV
│   │   └── preprocessor.py   # Text cleaning, PHI removal, filtering
│   ├── model/
│   │   └── trainer.py        # BART + LoRA fine-tuning pipeline
│   ├── evaluation/
│   │   └── metrics.py        # ROUGE, BERTScore, readability
│   └── app/
│       └── app.py            # Gradio web application
├── scripts/
│   ├── download_data.py      # Download and preprocess dataset
│   ├── train.py              # Start fine-tuning
│   ├── evaluate.py           # Run evaluation pipeline
│   └── predict.py            # Run inference on any text
├── tests/                    # Unit tests (all passing)
├── notebooks/                # Colab training notebook
├── docs/                     # Phase PDFs and evaluation results
├── app.py                    # HuggingFace Spaces entry point
├── config.yaml               # All hyperparameters
└── requirements.txt
```

## ⚙️ Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/medical-report-summariser.git
cd medical-report-summariser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
python src/app/app.py
# Open http://127.0.0.1:7860
```

## 🏋️ Training Your Own Model

```bash
# Step 1: Download and preprocess data
python scripts/download_data.py --max_samples 5000

# Step 2: Train (use Google Colab for GPU)
python scripts/train.py --smoke_test  # local test
# For real training: open notebooks/Medical_Report_Summariser_Training.ipynb on Colab

# Step 3: Evaluate
python scripts/evaluate.py --skip_bertscore

# Step 4: Run the app
python src/app/app.py
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | facebook/bart-large-cnn |
| Fine-tuning | LoRA (PEFT) — only 0.63% params trained |
| Training | HuggingFace Seq2SeqTrainer |
| Evaluation | ROUGE, BERTScore, Flesch-Kincaid |
| Web App | Gradio |
| Deployment | HuggingFace Spaces (free) |
| Data | ccdv/pubmed-summarization (PubMed abstracts) |

## 📈 How LoRA Works

Instead of retraining all 400M parameters of BART, LoRA adds tiny adapter layers to the attention matrices — only 884,736 parameters are trained (0.63%). Same quality, 100x less compute. Runs on a free Google Colab T4 GPU in ~30 minutes.

## 🔬 Evaluation

```bash
python scripts/evaluate.py --skip_bertscore
```

Sample output:
```
ROUGE-1   : 0.3392
ROUGE-2   : 0.1310
ROUGE-L   : 0.2708
Reduction : 61.5%  (117 words → 45 words)
Readability: 32.39 → 38.81 (+6.42 pts Flesch Ease)
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization) dataset
