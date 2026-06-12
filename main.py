import torch

from tokenizer import CharTokenizer
from train import train

CONFIG = {
    # Data
    "data_path": "data",
    # Model (MPS-friendly defaults; see README for CPU/GPU presets)
    "context_length": 256,
    "n_embd": 256,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.1,
    # Training
    "batch_size": 64,
    "epochs": 10,
    "lr": 3e-4,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = train(CONFIG, device)

    tokenizer = CharTokenizer.from_file(CONFIG["data_path"])
    prompt = "To be or not to be"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print("\n--- Generated text ---")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
