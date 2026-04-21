# Parameter Golf

OpenAI's challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Scored by bits-per-byte (BPB) on the FineWeb validation set — lower is better.

## Current Setup

- **GPUs**: 4xH100 (proxy for 8xH100 leaderboard runs)
- **Baseline BPB**: ~1.24 (4 GPU), leaderboard SOTA: ~1.11
- **Training script**: `train_gpt.py` (all hyperparams via env vars)
- **Data**: FineWeb dataset at `./data/datasets/fineweb10B_sp1024/`
- **Tokenizer**: `./data/tokenizers/fineweb_1024_bpe.model` (1024 vocab)

## Run Command (4 GPUs)

```bash
pip install sentencepiece && export RUN_ID=<name> && export DATA_PATH=./data/datasets/fineweb10B_sp1024/ && export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model && export VOCAB_SIZE=1024 && torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Override hyperparams with env vars (e.g. `NUM_LAYERS=10`, `TRAIN_SEQ_LEN=2048`, `MLP_MULT=3`).

## Key Constraints

- Compressed model (int8+zlib) + code must be under **16,000,000 bytes**
- Training capped at **10 minutes** wallclock (`MAX_WALLCLOCK_SECONDS=600`)
- Cannot access training data during evaluation

## Weights & Biases

- **Entity**: `anthony-chen-dev-zettabyte`
- **Project**: `parameter-golf`
- Enable with `WANDB=1` env var
- W&B MCP tools are available for querying runs, metrics, and history
