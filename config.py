from dataclasses import dataclass
from pathlib import Path

DATA_ROOT_PATH = Path("data")


@dataclass
class Config:
    # Data
    data_path: Path = DATA_ROOT_PATH / "data.txt"
    # Outputs
    output_dir: Path = Path("outputs")
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


@dataclass
class TestConfig(Config):
    # Data
    data_path: Path = DATA_ROOT_PATH / "test.txt"
    # Model (MPS-friendly defaults; see README for CPU/GPU presets)
    context_length: int = 2
    n_embd: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
