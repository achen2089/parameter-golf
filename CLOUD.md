# Cloud Compute — Rapid Dev Reference

How to launch training jobs on the remote GPU platform.

The baseline train_gpt.py is now the **PR #1394 SP8192 stack** (ported from the 2026-04-05 submission, ~1.0856 BPB target on 8×H100). Hyperparam defaults are tuned for that recipe — just set env vars to override.

## Prereq: SP8192 Data (one-time per cloud machine)

The current `train_gpt.py` expects `./data/datasets/fineweb10B_sp8192/` and `./data/tokenizers/fineweb_8192_bpe.model`. If you're on a fresh cloud machine, add this to the install step of Master Command (it's a ~4-5 GB download, one-time):

```bash
[[ -d ./data/datasets/fineweb10B_sp8192 ]] || \
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

## Master Command

Use **Custom Mode**:

```bash
/bin/bash
-c
git pull --ff-only && \
pip install -q sentencepiece wandb brotli && \
{ [[ -d ./data/datasets/fineweb10B_sp8192 ]] || \
  { rm -f data/manifest.json data/datasets/manifest.json && \
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128 ; } ; } && \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

The `rm -f data/manifest.json` step is important: the default `willdepueoai/parameter-golf` repo doesn't have SP8192, so we need the manifest from `kevclark/parameter-golf`. If an old manifest is cached from a prior SP1024 download, the script will skip re-fetching it and fail with `dataset fineweb10B_sp8192 not found in datasets/manifest.json`. Clearing the manifest forces a fresh pull from the correct repo.

**`git pull --ff-only` is critical** — without it, a duplicated job will run whatever code was on the machine at the time the snapshot was taken, not your latest commit. `--ff-only` refuses to merge if local diverges, which protects against accidents.

### IDE convenience: `./main` and `./base`

When SSH'd into the box (or running inside the cloud IDE), you can skip the Master Command boilerplate and just run:

- `./main` — runs `train_gpt.py` (SP8192 Phase N recipe). Auto-detects GPU count, downloads SP8192 data if missing, sets W&B defaults.
- `./base` — runs `base_train_gpt.py` (SP1024 baseline, A/B control). Same treatment for SP1024 data.

Both scripts honor `RUN_ID`, `NPROC`, any train_gpt.py env var, and pass-through overrides:

```bash
./main p1_sp8192_base               # positional: RUN_ID
RUN_ID=p2_qkg5 QK_GAIN_INIT=5.0 ./main
NPROC=8 ./main
```

These are **not** a replacement for the Master Command on the platform — the platform still needs the full `git pull && pip install && ...` form above. The scripts are just a shortcut once you're interactive on the machine.

- `--nproc_per_node=4` → 4 GPUs. Change to `8` for legal competition-spec runs.
- `--standalone` → single-node distributed.
- Script auto-adjusts: `grad_accum_steps = 8 // world_size`, so 4 GPUs is mathematically equivalent to 8 (just ~2× slower wallclock).

## Environment Variables

Set these in the platform's Environment Variables form. Per-run overrides only — everything else uses sane defaults in `train_gpt.py`.

### Required (always set these)

| Variable | Value | Notes |
|----------|-------|-------|
| `RUN_ID` | `p1_sp8192_base` | **Change per run** — W&B run name + log filename |
| `WANDB` | `1` | Enable W&B logging |
| `WANDB_API_KEY` | `<your key>` | Do NOT commit |
| `WANDB_ENTITY` | `anthony-chen-dev-zettabyte` | W&B entity |
| `WANDB_PROJECT` | `parameter-golf` | W&B project |

Data/tokenizer paths are auto-derived from `VOCAB_SIZE` (default 8192) — no need to set `DATA_PATH`/`TOKENIZER_PATH` anymore.

### Core knobs (defaults match 1.0856 BPB target)

| Variable | Default | Notes |
|----------|---------|-------|
| `VOCAB_SIZE` | `8192` | Must match installed tokenizer |
| `NUM_LAYERS` | `11` | SOTA uses 11 |
| `MODEL_DIM` | `512` | Width |
| `NUM_HEADS` / `NUM_KV_HEADS` | `8` / `4` | GQA |
| `MLP_MULT` | `4.0` | MLP expansion |
| `TRAIN_SEQ_LEN` | `2048` | Default doubled vs old baseline |
| `EVAL_SEQ_LEN` | `2048` | Match training |
| `TRAIN_BATCH_TOKENS` | `786432` | 2048 × 48 × 8 |
| `ITERATIONS` | `20000` | Capped by wallclock |
| `MAX_WALLCLOCK_SECONDS` | `600` | 10-min cap |

### Architecture levers (introduced by PR #1394 stack)

| Variable | Default | What it does |
|----------|---------|--------------|
| `ROPE_DIMS` | `16` | Partial RoPE: rotate first 16 of 64 head dims |
| `QK_GAIN_INIT` | `4.0` | Learnable per-head QK scale. P2 → 5.0, P5 → 5.25 |
| `XSA_LAST_N` | `11` | Exclusive Self-Attention across all layers |
| `LN_SCALE` | `1` | Per-layer `1/sqrt(layer+1)` norm damping |
| `SKIP_GATES_ENABLED` | `1` | Sigmoid-gated U-Net skip connections |
| `LOOP_START` / `LOOP_END` | `4` / `5` | Depth recurrence layer range |
| `NUM_LOOPS` | `2` | Times to repeat looped layers |
| `ENABLE_LOOPING_AT` | `0.5` | Fraction of training to activate recurrence |
| `SLIDING_WINDOW_ENABLED` | `1` | Sliding-window val eval (stride 64) |

### Optimizer / schedule

| Variable | Default | Notes |
|----------|---------|-------|
| `MATRIX_LR` | `0.02` | Muon LR |
| `EMBED_LR` | `0.6` | Adam on tok_emb |
| `TIED_EMBED_LR` | `0.03` | Tied-embedding LR |
| `MUON_MOMENTUM` | `0.99` | Muon momentum |
| `MUON_ROW_NORMALIZE` | `1` | MuonEq-R variant |
| `MUON_WD` | `0.085` | Muon weight decay |
| `WARMDOWN_FRAC` | `0.667` | Fraction of training for LR warmdown |
| `EMA_DECAY` | `0.997` | EMA weight averaging decay |
| `GRAD_CLIP_NORM` | `0.3` | Gradient clipping |

### Quantization / compression

| Variable | Default | Notes |
|----------|---------|-------|
| `COMPRESSOR` | `brotli` | `brotli`/`lzma`/`zlib`. Brotli-11 is current SOTA |
| `MATRIX_BITS` | `6` | Int6 for MLP/attn weights |
| `EMBED_BITS` | `8` | Int8 for embeddings |
| `MATRIX_CLIP_SIGMAS` | `12.85` | SDClip k for matrices |
| `EMBED_CLIP_SIGMAS` | `20.0` | SDClip k for embeddings |
| `GPTQ_CALIBRATION_BATCHES` | `64` | Self-gen calibration batches for GPTQ |

## Phase Runbook

See full roadmap at `~/.claude/plans/silly-launching-salamander.md`. Quick version:

| Phase | RUN_ID | Overrides | Target BPB |
|-------|--------|-----------|-----------:|
| P1 | `p1_sp8192_base` | (all defaults) | ~1.0856 |
| P2 | `p2_qkg5` | `QK_GAIN_INIT=5.0` | ~1.0828 |
| P3 | `p3_ttt` | P2 + `TTT_ENABLED=1` (needs code port) | ~1.0825 |
| P4 | `p4_parres` | P3 + parallel residuals (needs code port) | ~1.0822 |
| P5 | `p5_sota_arch` | `QK_GAIN_INIT=5.25 LOOP_START=3 LOOP_END=5 NUM_LOOPS=3 ENABLE_LOOPING_AT=0.35 WARMDOWN_FRAC=0.72` | ~1.0815 |
| P6 | `p6_sota_s{42,314,999}` | P5 + `MUON_WD=0.095 MATRIX_LR=0.022 EMA_DECAY=0.9965 SEED=42/314/999` | ~1.0810 |

## Rapid Dev Loop

1. Edit `train_gpt.py` locally → commit → push.
2. Duplicate previous cloud job, bump `RUN_ID`, tweak env vars.
3. Launch → watch W&B at `https://wandb.ai/anthony-chen-dev-zettabyte/parameter-golf`.
4. Compare runs in W&B (filter by config, plot `val_bpb` curves).

## Submission Size Budget

- Hard limit: **16,000,000 bytes** total (code + compressed weights).
- Watch `submission_bytes` in W&B summary.
- If over budget: tighten `MATRIX_CLIP_SIGMAS` (lower = smaller), drop `NUM_LAYERS`, or reduce `MODEL_DIM` — **not** `MAX_WALLCLOCK_SECONDS`.

## GPU Count and Wallclock Auto-Scaling

The `grad_accum_steps = 8 // world_size` math makes each **step** mathematically equivalent between 4-GPU and 8-GPU runs. But under a fixed 600s wallclock, 4 GPUs produce only **half the optimizer updates** → undertrained model → worse BPB.

**Observed**: first `p1_sp8192_base` run on 4 GPUs at 600s got 2,644 steps / 1.1522 BPB vs ~6,900 steps / 1.0856 BPB expected on 8 GPUs. `train_loss` was still descending at the cap.

**Fix**: `./main` and `./base` now auto-scale `MAX_WALLCLOCK_SECONDS` by NPROC:

| NPROC | MAX_WALLCLOCK_SECONDS | Real wallclock | Step count equivalent to |
|-------|----------------------:|---------------:|--------------------------|
| 8 | 600 | 10 min | 8×H100 legal submission |
| 4 | 1200 | 20 min | 8×H100 legal submission |
| 2 | 2400 | 40 min | 8×H100 legal submission |
| 1 | 4800 | 80 min | 8×H100 legal submission |

Explicitly setting `MAX_WALLCLOCK_SECONDS` in env overrides auto-scaling. For **legal competition submissions** you must run on 8×H100 at 600s — these scaled proxies are quality-equivalent for iteration only.

## Known Gotchas
- **Flash Attention 3** is optional — script falls back to SDPA if `flash-attn-interface` isn't installed. Install path: `pip install flash-attn --no-build-isolation` (slow, ~10 min).
- **Don't mix SP1024 and SP8192 runs** without confirming `VOCAB_SIZE` matches the tokenizer file on disk.
- The reference `train_gpt.py` files in `records/track_10min_16mb/*/train_gpt.py` are LZMA+base85 compressed one-liners (code size counts toward 16MB budget). Decompress with `lzma.decompress(base64.b85decode(...))` if you need to read them.
