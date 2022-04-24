from torch import Tensor
from torch.nn import Module, \
    ModuleList
from typing import Tuple

from .EncoderLayer import EncoderLayer


class Encoder(Module):
    def __init__(
            self,
            sequence_length: int,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            n_heads: int = 2,
            embed_dim: int = 8,
            num_layers: int = 2,
            dropout: float = 0.1
    ) -> None:
        super(Encoder, self).__init__()

        self.layers = ModuleList([
            EncoderLayer(sequence_length, input_shape, n_heads, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def forward(self, tokens: Tensor, mask: Tensor = None) -> Tensor:
        out = None
        for layer in self.layers:
            out = layer(tokens, mask)

        return out
