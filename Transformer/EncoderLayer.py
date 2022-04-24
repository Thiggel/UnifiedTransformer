from torch import Tensor
from torch.nn import Module, \
    Sequential, \
    Linear, \
    LayerNorm, \
    ReLU, \
    Dropout
from typing import Tuple

from .MultiHeadAttention import MultiHeadAttention


class EncoderLayer(Module):
    def __init__(
            self,
            sequence_length: int,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            n_heads: int = 2,
            embed_dim: int = 8,
            dropout: float = 0.1
    ) -> None:
        super(EncoderLayer, self).__init__()

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        self.input_shape = input_shape

        self.norm1 = LayerNorm([sequence_length, embed_dim])

        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm2 = LayerNorm([sequence_length, embed_dim])

        self.MLP = Sequential(
            Linear(embed_dim, embed_dim),
            ReLU()
        )

        self.dropout = Dropout(dropout)

    def forward(self, tokens: Tensor, mask: Tensor = None) -> Tensor:
        out = tokens + self.attention(self.norm1(tokens), mask)

        out = out + self.MLP(self.norm2(out))

        return self.dropout(out)
