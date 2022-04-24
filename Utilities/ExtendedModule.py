from torch.nn import Module
from torch import Tensor


class ExtendedModule(Module):
    def __init__(self):
        super().__init__()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        pass
