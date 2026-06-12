from dataclasses import dataclass
from pathlib import Path

import torch

from report import write_report
from tokenizer import CharTokenizer
from train import train


@dataclass
class Config:
    # Data
    data_path: Path = Path("data.txt")
    # Model (MPS-friendly defaults; see README for CPU/GPU presets)
    context_length: int = 256
    n_embd: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    # Training
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4


CONFIG = Config()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model, history = train(CONFIG, device)

    tokenizer = CharTokenizer.from_file(CONFIG.data_path)
    write_report(history, CONFIG, model, tokenizer, device)


if __name__ == "__main__":
    main()
