# Agent Instructions

## Project Context

This is the OpenAI Parameter Golf challenge. The goal is to minimize val_bpb (bits per byte) on the FineWeb validation set with a model that fits in 16MB compressed and trains in under 10 minutes on 8xH100s.

## Weights & Biases

- **Entity**: `anthony-chen-dev-zettabyte`
- **Project**: `parameter-golf`
- Use the W&B MCP tools (`mcp__wandb__*`) to query run metrics, compare experiments, and track progress
- Key metrics to track: `val_bpb`, `val_loss`, `train_loss`, `lr_scale`
- Final metrics in run summary: `final_val_bpb`, `final_val_loss`, `submission_bytes`, `training_time_ms`, `total_steps`
- Enable W&B logging by setting `WANDB=1` in env vars

## Experiment Workflow

1. All hyperparams are controlled via environment variables (see `Hyperparameters` class in `train_gpt.py`)
2. Always set a descriptive `RUN_ID` for each experiment
3. After a run, check both BPB score AND compressed model size — must be under 16MB
4. Use 4 GPUs for iteration (`--nproc_per_node=4`), final submissions need 8 GPUs
5. Compare runs in W&B to track what changes helped

## Key Leaderboard Techniques (in order of impact)

1. **Int6 quantization (QAT)** — fits ~50% more params in 16MB (biggest win)
2. **More layers** (10-11) + **3x MLP** — better architecture
3. **Sliding window eval** — free BPB improvement at eval time
4. **EMA/SWA** — smoother final model weights
5. **Weight decay on Muon** (WD=0.04)
6. **BigramHash** — cheap bigram lookup table
7. **XSA** — cross-sequence attention during eval
8. **GPTQ** — smarter post-training quantization
9. **TTT** — test-time training on already-scored val data
