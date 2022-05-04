from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor
from .MnistMultipleNumbers import MnistTrueFalseMultipleNumbers
from typing import Optional

from .DataModule import DataModule


class MnistDataModule(DataModule):
    def __init__(self, batch_size: Optional[int] = 32, fashion_mnist: bool = False) -> None:
        transform = ToTensor()
        root = './datasets'

        mnist_full = MnistTrueFalseMultipleNumbers(
            root=root,
            train=True,
            download=True,
            transform=transform,
            fashion_mnist=fashion_mnist
        )

        mnist_len = len(mnist_full)
        train_length = int(mnist_len * 0.7)
        val_length = mnist_len - train_length

        self.train_set, self.val_set = random_split(
            mnist_full, [train_length, val_length]
        )

        self.test_set = MnistTrueFalseMultipleNumbers(
            root=root,
            train=False,
            download=True,
            transform=transform,
            fashion_mnist=fashion_mnist
        )

        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size)
