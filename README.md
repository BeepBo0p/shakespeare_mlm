# Shakespeare - Medium Language Model (MLM)

Character-level autoregressive transformer trained on the Complete Works of Shakespeare.

## Download dataset

```bash
curl https://www.gutenberg.org/cache/epub/100/pg100.txt # Thank you to project gutenberg for being the best
```

## Setup

```bash
uv sync
```

## Usage

```bash
python main.py
```

Adjust `CONFIG` in `main.py` to match your hardware:

| Target      | n_embd | n_heads | n_layers | context_length | ~Params |
|-------------|--------|---------|----------|----------------|---------|
| CPU (test)  | 64     | 2       | 2        | 128            | 0.2M    |
| MacBook MPS | 256    | 8       | 6        | 256            | 4.8M    |
| GPU (CUDA)  | 512    | 8       | 8        | 512            | 25M     |

Device is auto-selected: CUDA > MPS > CPU.

## Output files

| File | Description |
|------|-------------|
| `ckpt_{epoch}.pt` | Model checkpoint saved after each epoch |
| `training_log.csv` | Per-epoch metrics: loss, lr, time |
| `training_report.md` | Full post-training summary with generated sample |

## File structure

| File | Role |
|------|------|
| `tokenizer.py` | Character-level tokenizer with ASCII normalisation |
| `dataset.py` | PyTorch `Dataset` for sliding-window context windows |
| `model.py` | GPT model: causal self-attention, pre-LN transformer blocks, weight tying |
| `train.py` | Training loop, evaluation, per-epoch CSV logging |
| `report.py` | Post-training report generator |
| `main.py` | Entry point and config |
