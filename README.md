# AG News Classification with Transformers 📰

Comparative study of three transformer models (RoBERTa, DeBERTa, and ModernBERT) for news classification on the AG News dataset.

## 📋 Project Overview

This project trains and compares three state-of-the-art transformer models on the AG News dataset for multi-class text classification:

- **RoBERTa** (roberta-base)
- **DeBERTa v3** (microsoft/deberta-v3-base)
- **ModernBERT** (answerdotai/ModernBERT-base)

### Dataset

**AG News** contains news articles from more than 2000 sources, classified into 4 categories:
- 0: World
- 1: Sports
- 2: Business
- 3: Science/Technology

- **Training samples**: 120,000
- **Test samples**: 7,600

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ag-news-classification.git
cd ag-news-classification

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Run in Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`
3. Run all cells

#### Option 2: Run locally

```bash
python train.py
```

**Note:** A GPU is strongly recommended for faster training.

## 📊 Data Split

- **Training**: 70% (14,000 samples from subset)
- **Validation**: 15% (1,500 samples from subset)
- **Test**: 15% (1,500 samples from subset)

The code uses a subset of 20,000 training and 3,000 test samples for faster experimentation. To use the full dataset, modify lines 30-31 in the code.

## 🎯 Results

The models are evaluated on:
- **Accuracy**
- **F1-Score** (weighted)
- **Precision** (weighted)
- **Recall** (weighted)

Results are visualized in two plots:
1. F1-Score comparison bar chart
2. Grouped bar chart with all metrics

### Example Results

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| RoBERTa | 0.9XXX | 0.9XXX | 0.9XXX | 0.9XXX |
| DeBERTa | 0.9XXX | 0.9XXX | 0.9XXX | 0.9XXX |
| ModernBERT | 0.9XXX | 0.9XXX | 0.9XXX | 0.9XXX |

## 🛠️ Technical Details

### Model Configuration

- **Max sequence length**: 128 tokens
- **Batch size**: 16 (training), 32 (evaluation)
- **Epochs**: 2
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning rate**: 5e-5 (default)
- **Warmup steps**: 200
- **FP16**: Enabled (if GPU available)

### Training Time

Approximate training time on Google Colab with T4 GPU:
- Per model: ~8-10 minutes
- Total (3 models): ~25-30 minutes

## 📁 Project Structure

```
ag-news-classification/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── results/             # Model checkpoints (generated)
│   ├── roberta-base/
│   ├── deberta-v3-base/
│   └── ModernBERT-base/
└── logs/                # Training logs (generated)
```

## 📈 Visualizations

The script generates two plots:

1. **F1-Score Comparison**: Bar chart comparing F1-scores across models
2. **All Metrics Comparison**: Grouped bar chart showing accuracy, F1-score, precision, and recall

Plots are saved as:
- `f1_score_comparison.png`
- `all_metrics_comparison.png`

## 🔧 Customization

### Use Full Dataset

```python
# Comment out these lines (30-31)
# train_df = train_df.sample(n=20000, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(n=3000, random_state=42).reset_index(drop=True)
```

### Change Hyperparameters

Modify the `TrainingArguments` in the `train_model` function:

```python
args = TrainingArguments(
    num_train_epochs=3,                    # Change number of epochs
    per_device_train_batch_size=32,        # Adjust batch size
    learning_rate=3e-5,                     # Change learning rate
    # ... other parameters
)
```

### Add More Models

```python
models = {
    'RoBERTa': 'roberta-base',
    'DeBERTa': 'microsoft/deberta-v3-base',
    'ModernBERT': 'answerdotai/ModernBERT-base',
    'BERT': 'bert-base-uncased',            # Add new model
}
```

## 💾 Saving Models

To save the best model:

```python
# After training
trainer.save_model('./best_model')
tokenizer.save_pretrained('./best_model')
```

## 🤝 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 5GB+ disk space

## 📝 License

MIT License

## 🙏 Acknowledgments

- **AG News Dataset**: [Original source](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- **HuggingFace**: For the Transformers library and model hub
- **Model creators**: 
  - RoBERTa by Facebook AI
  - DeBERTa by Microsoft
  - ModernBERT by Answer.AI

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔗 Useful Links

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [AG News Dataset on HuggingFace](https://huggingface.co/datasets/fancyzhx/ag_news)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

⭐ If you find this project useful, please consider giving it a star!
