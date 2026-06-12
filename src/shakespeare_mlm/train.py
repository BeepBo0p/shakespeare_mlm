import csv
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import ShakespeareDataset
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer

_GPT_CONFIG_FIELDS = {"context_length", "n_embd", "n_heads", "n_layers", "dropout"}
_CSV_FIELDS = ["epoch", "train_loss", "val_loss", "lr", "epoch_secs"]


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        if count >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def train(config: Config, device: torch.device) -> tuple[GPT, list[dict]]:
    tokenizer = CharTokenizer.from_file(config.data_path)
    tokens = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)

    n = int(0.9 * len(tokens))
    train_ds = ShakespeareDataset(tokens[:n], config.context_length)
    val_ds = ShakespeareDataset(tokens[n:], config.context_length)

    print(len(train_ds))
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=pin, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=pin, num_workers=0
    )

    config_as_dict = config.__dict__

    gpt_kwargs = {k: config_as_dict[k] for k in _GPT_CONFIG_FIELDS if k in config_as_dict}
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **gpt_kwargs)
    model = GPT(cfg).to(device)
    print(f"Model: {model.num_params():,} parameters | vocab_size={cfg.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs
    )

    history: list[dict] = []

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(config.output_dir / "training_log.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()

        for epoch in range(config.epochs):
            model.train()
            epoch_start = time.perf_counter()
            train_total, train_steps = 0.0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_total += loss.item()
                train_steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            val_loss = evaluate(model, val_loader, device)
            epoch_secs = time.perf_counter() - epoch_start
            train_loss = train_total / train_steps
            lr = scheduler.get_last_lr()[0]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "epoch_secs": epoch_secs,
            }
            history.append(row)
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})
            csv_file.flush()

            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  time={epoch_secs:.1f}s")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_dir / f"ckpt_{epoch}.pt",
            )

    return model, history
