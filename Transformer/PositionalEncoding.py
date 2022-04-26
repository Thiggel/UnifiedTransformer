from numpy import sin, cos
import torch
from torch import Tensor, ones, device, cuda
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, sequence_length):
        super(PositionalEncoding, self).__init__()

        self.dev = device("cuda:0" if cuda.is_available() else "cpu")

        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

        self.positional_encoding = self.get_positional_embeddings()

    def get_positional_embeddings(self) -> Tensor:
        result = ones(self.sequence_length, self.embed_dim)

        for i in range(self.sequence_length):
            for j in range(self.embed_dim):
                result[i][j] = sin(i / (10000 ** (j / self.embed_dim))) \
                    if j % 2 == 0 \
                    else cos(i / (10000 ** ((j - 1) / self.embed_dim)))

        return result

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.positional_encoding.repeat(x.shape[0], 1, 1).to(self.dev)

        return x
