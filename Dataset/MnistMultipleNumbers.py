from math import sqrt
from isqrt import isqrt
from torch.utils.data import Dataset
from torch import cat, Tensor, zeros
from typing import Callable, Optional, List, Union, Tuple
from torchvision.datasets.mnist import MNIST
from random import randint


class MnistMultipleNumbers(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = True,
            transform: Optional[Callable] = None,
            num_digits_per_picture: int = 4
    ) -> None:
        super().__init__()

        assert num_digits_per_picture == isqrt(num_digits_per_picture) ** 2, \
            "The number of digits per picture must be a perfect square"

        self.num_digits_per_picture = num_digits_per_picture
        self.depth = int(sqrt(self.num_digits_per_picture))

        self.mnist = MNIST(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        images: List[Union[Tensor, None]] = [None] * self.depth
        targets = zeros((self.num_digits_per_picture,))

        for row_idx in range(self.depth):
            current_row: List[Union[Tensor, None]] = [None] * self.depth
            for col_idx in range(self.depth):
                image, target = self.mnist[randint(0, len(self.mnist) - 1)]

                current_row[col_idx] = image
                targets[row_idx * col_idx] = target

            images[row_idx] = cat(tuple(current_row), dim=2)

        return cat(tuple(images), dim=1), targets

    def __len__(self) -> int:
        return len(self.mnist)


class MnistTrueFalseMultipleNumbers(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = True,
            transform: Optional[Callable] = None,
            num_digits_per_picture: int = 4
    ) -> None:
        super().__init__()

        self.mnist = MnistMultipleNumbers(
            root=root,
            train=train,
            download=download,
            transform=transform,
            num_digits_per_picture=num_digits_per_picture
        )

    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        image, numbers = self.mnist[index]
        target = 1

        if randint(0, 1) == 0:
            numbers[randint(0, len(numbers) - 1)] = randint(0, 9)
            target = 0

        return (image, numbers), target

    def __len__(self) -> int:
        return len(self.mnist)
