import torch

from config import Config, TestConfig
from report import write_report
from tokenizer import CharTokenizer
from train import train

TEST = True

CONFIG = Config() if not TEST else TestConfig()


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
    write_report(
        history, CONFIG, model, tokenizer, device, CONFIG.output_dir / "training_report.md"
    )


if __name__ == "__main__":
    main()
