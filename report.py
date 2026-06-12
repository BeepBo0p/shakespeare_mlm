from datetime import datetime

import torch

from main import Config
from model import GPT
from tokenizer import CharTokenizer


def write_report(
    history: list[dict],
    config: Config,
    model: GPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    output_path: str = "training_report.md",
) -> None:
    cfg = model.config
    n_tokens = len(tokenizer.text)
    n_train = int(0.9 * n_tokens)
    n_val = n_tokens - n_train
    total_secs = sum(r["epoch_secs"] for r in history)
    best = min(history, key=lambda r: r["val_loss"])

    prompt = "To be or not to be"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    sample = tokenizer.decode(out[0].tolist())

    lines: list[str] = []

    def h(text: str) -> None:
        lines.append(text)

    h("# Training Report\n")
    h(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    h(f"**Device:** {device}  ")
    h(f"**Total training time:** {_fmt_duration(total_secs)}\n")

    h("## Dataset\n")
    h("| | |")
    h("|---|---|")
    h(f"| Corpus characters | {n_tokens:,} |")
    h(f"| Vocab size | {cfg.vocab_size} |")
    h(f"| Train tokens | {n_train:,} |")
    h(f"| Val tokens | {n_val:,} |")
    h("")

    h("## Model\n")
    h("| | |")
    h("|---|---|")
    h(f"| Parameters | {model.num_params():,} |")
    h(f"| context_length | {cfg.context_length} |")
    h(f"| n_embd | {cfg.n_embd} |")
    h(f"| n_heads | {cfg.n_heads} |")
    h(f"| n_layers | {cfg.n_layers} |")
    h(f"| dropout | {cfg.dropout} |")
    h("")

    h("## Training config\n")
    h("| | |")
    h("|---|---|")
    h(f"| batch_size | {config.batch_size} |")
    h(f"| epochs | {config.epochs} |")
    h(f"| peak lr | {config.lr:.2e} |")
    h("")

    h("## Metrics\n")
    h("| epoch | train_loss | val_loss | Δval | time(s) |")
    h("|------:|----------:|---------:|-----:|--------:|")
    prev_val = None
    for r in history:
        delta = f"{r['val_loss'] - prev_val:+.4f}" if prev_val is not None else "—"
        row = f"| {r['epoch']} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | {delta} | {r['epoch_secs']:.1f} |"
        if r["epoch"] == best["epoch"]:
            row = f"| **{r['epoch']}** | **{r['train_loss']:.4f}** | **{r['val_loss']:.4f}** | {delta} | {r['epoch_secs']:.1f} |"
        h(row)
        prev_val = r["val_loss"]
    h("")

    h(
        f"**Best checkpoint:** epoch {best['epoch']} — val_loss={best['val_loss']:.4f} → `ckpt_{best['epoch']}.pt`\n"
    )

    h("## Generated sample\n")
    h(f'Prompt: `"{prompt}"`\n')
    h("```")
    h(sample)
    h("```")

    report = "\n".join(lines)
    print("\n" + report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport written to {output_path}")


def _fmt_duration(secs: float) -> str:
    secs = int(secs)
    h, remainder = divmod(secs, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
