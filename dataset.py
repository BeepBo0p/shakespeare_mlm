from torch import Tensor
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, tokens: Tensor, context_length: int) -> None:
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]
