from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.itos[i] for i in indices)

    @classmethod
    def from_file(cls, path: Path, normalise: bool = True) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if normalise:
            text = (
                text.replace("'", "'")
                .replace("'", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("—", "-")
                .replace("\t", " ")
            )
            text = "".join(c for c in text if ord(c) < 128)
        return cls(text)
