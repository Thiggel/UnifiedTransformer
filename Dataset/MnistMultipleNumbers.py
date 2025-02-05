from math import sqrt
from isqrt import isqrt
from torch.utils.data import Dataset
from torch import cat, Tensor, zeros, tensor
from typing import Callable, Optional, List, Union, Tuple
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import FashionMNIST
from random import randint
from torch.nn.functional import one_hot


class MnistMultipleNumbers(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = True,
            transform: Optional[Callable] = None,
            num_digits_per_picture: int = 4,
            fashion_mnist: bool = False
    ) -> None:
        super().__init__()

        assert num_digits_per_picture == isqrt(num_digits_per_picture) ** 2, \
            "The number of digits per picture must be a perfect square"

        self.num_digits_per_picture = num_digits_per_picture
        self.depth = int(sqrt(self.num_digits_per_picture))

        dataset_type = MNIST if not fashion_mnist else FashionMNIST

        self.mnist = dataset_type(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        images: List[Union[Tensor, None]] = [None] * self.depth
        targets = zeros((self.num_digits_per_picture,))

        for row_idx in range(self.depth):
            current_row: List[Union[Tensor, None]] = [None] * self.depth
            for col_idx in range(self.depth):
                image, target = self.mnist[randint(0, len(self.mnist) - 1)]

                current_row[col_idx] = image
                targets[row_idx * self.depth + col_idx] = target

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
            num_digits_per_picture: int = 4,
            fashion_mnist: bool = False
    ) -> None:
        super().__init__()

        self.mnist = MnistMultipleNumbers(
            root=root,
            train=train,
            download=download,
            transform=transform,
            num_digits_per_picture=num_digits_per_picture,
            fashion_mnist=fashion_mnist
        )

    @staticmethod
    def randomize_number(num: int) -> int:
        return (num + randint(0, 9)) % 10

    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        image, numbers = self.mnist[index]
        target = 1

        # two outputs for true/false + 10 outputs for the individual numbers
        number_target = one_hot(numbers.long() + 2, num_classes=12).sum(0).clamp(max=1)

        if randint(0, 1) == 0:
            randIdx = randint(0, len(numbers) - 1)
            numbers[randIdx] = self.randomize_number(numbers[randIdx])

            # all other numbers have 1/4 chance of also being wrong
            for idx in range(len(numbers)):
                if idx != randIdx and randint(0, 3) == 0:
                    numbers[idx] = self.randomize_number(numbers[idx])

            target = 0

        target = one_hot(tensor(target), num_classes=12) + number_target

        return (image, numbers), target

    def __len__(self) -> int:
        return len(self.mnist)
