# Ultra Nano GPT

A minimal GPT-2 implementation for learning and experimentation, inspired by Andrej Karpathy's nanoGPT.

## Project Overview

This project implements a simplified version of GPT-2 from scratch using PyTorch. It's designed for educational purposes to understand the architecture and training process of transformer-based language models.

## Project Structure

```
ultra-nano-gpt/
├── my_gpt/
│   ├── model.py          # GPT model architecture
│   ├── train.py          # Training script
│   ├── generate.py       # Text generation script
│   ├── my_gpt.py         # Main GPT implementation
│   ├── my_gpt_v1.py      # Earlier version
│   └── input.txt         # Training data
├── input.txt             # Training data (root level)
├── test.ipynb            # Jupyter notebook for testing
├── .venv/                # Virtual environment
├── .gitignore            # Git ignore rules
└── CLAUDE.md             # This file
```

## Key Components

### Model Architecture (`model.py`)
- Implements GPT-2 architecture with multi-head self-attention
- Supports loading pretrained GPT-2 weights
- Configurable model size (embedding dimension, number of heads, layers, etc.)

### Training (`train.py`)
- Character-level tokenization
- AdamW optimizer with learning rate scheduling
- Checkpoint saving functionality
- Supports training on CPU, CUDA, or MPS (Apple Silicon)

### Generation (`generate.py`)
- Text generation using trained or pretrained models
- Top-k sampling for diverse outputs
- Supports both custom-trained and pretrained GPT-2 models
- Uses tiktoken for GPT-2 tokenization (when available)

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install torch numpy tiktoken
   ```

3. **Prepare training data:**
   - Place your text data in `input.txt`
   - The model will learn character-level patterns from this data

## Usage

### Training a Model

```bash
cd my_gpt
python train.py
```

This will:
- Load and process `input.txt`
- Train a GPT model with the configured parameters
- Save checkpoints to `ckpt.pt`
- Display training progress and loss

### Generating Text

**Using pretrained GPT-2:**
```bash
cd my_gpt
python generate.py
```

**Using your trained model:**
- Edit `generate.py` and set `load_pretrained = False`
- Ensure `ckpt.pt` exists from training
- Run: `python generate.py`

## Configuration

### Model Configuration (in `model.py`)
```python
GPTConfig(
    block_size=256,    # Maximum sequence length
    vocab_size=...,    # Vocabulary size (auto-detected)
    n_layer=6,         # Number of transformer blocks
    n_head=6,          # Number of attention heads
    n_embd=384,        # Embedding dimension
    dropout=0.2        # Dropout rate
)
```

### Training Configuration (in `train.py`)
- Batch size: 64
- Learning rate: 3e-4
- Max iterations: 5000
- Evaluation interval: 500 steps

## Device Support

The code automatically detects and uses the best available device:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback option

## Features

- ✅ GPT-2 architecture implementation
- ✅ Pretrained model loading
- ✅ Custom training on your own data
- ✅ Character-level tokenization
- ✅ Top-k sampling for generation
- ✅ Checkpoint saving/loading
- ✅ Multi-device support (CPU/CUDA/MPS)

## Learning Resources

This implementation is based on:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT implementation

## Next Steps

1. **Experiment with hyperparameters**: Try different model sizes, learning rates, and batch sizes
2. **Use different datasets**: Train on various text corpora to see how the model adapts
3. **Implement improvements**: Add features like gradient accumulation, mixed precision training, or better sampling strategies
4. **Scale up**: Try larger models with more layers and parameters
5. **Fine-tuning**: Load pretrained GPT-2 and fine-tune on specific tasks

## Notes

- The model uses character-level tokenization for custom training (simpler but less efficient than BPE)
- Pretrained GPT-2 uses tiktoken for proper tokenization
- Training checkpoints are saved as `ckpt.pt`
- The generation script includes error handling for decoding issues

## Troubleshooting

**Issue**: `tiktoken not installed`
- **Solution**: `pip install tiktoken` or use custom tokenization

**Issue**: Out of memory during training
- **Solution**: Reduce batch size or model size in the config

**Issue**: `ckpt.pt not found`
- **Solution**: Train a model first or set `load_pretrained = True`

---

*This is a learning project. For production use, consider using established libraries like Hugging Face Transformers.*
