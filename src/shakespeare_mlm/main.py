import torch

from .config import TestConfig, make_config
from .report import write_report
from .tokenizer import CharTokenizer
from .train import train

TEST = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    config = TestConfig() if TEST else make_config(device)
    model, history = train(config, device)

    tokenizer = CharTokenizer.from_file(config.data_path)
    write_report(
        history, config, model, tokenizer, device, config.output_dir / "training_report.md"
    )


if __name__ == "__main__":
    main()
