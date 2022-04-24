from torch import Tensor
from torch.nn import Module, \
    Linear, \
    Sequential, \
    Conv3d, \
    ReLU
from typing import Tuple


class PatchEmbedding(Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            embed_dim: int = 8,
            conv_layers: int = 0,
    ):
        super(PatchEmbedding, self).__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size

        self.kernel_size = 3
        self.conv_layers = conv_layers

        self.conv = Sequential(*[
            Sequential(
                Conv3d(
                    self.get_layer_channels(index),
                    self.get_layer_channels(index + 1),
                    kernel_size=(1, self.kernel_size, self.kernel_size),
                    stride=(1, 1, 1)
                ),
                ReLU()
            )
            for index in range(conv_layers)
        ])

        self.linear_projection = Linear(self.input_dim, embed_dim)

    def get_layer_channels(self, layer: int) -> int:
        if layer == 0 or self.conv_layers == 0:
            return self.input_shape[0]

        if layer == -1:
            layer = self.conv_layers

        assert layer <= self.conv_layers, "Tried to calculate channel count for inexisting conv layer"

        return 16 * layer

    @property
    def reduced_patch_size(self) -> Tuple[int, int]:
        """
        calculate patch size (which will be different after
        convolution), acc. to formula:
        (h, w) -> (h - kernel_size + 1, w - kernel_size + 1)
        :return: The new patch size after convolution all patches
        """
        conv_size_reduction = (-self.kernel_size + 1) * self.conv_layers
        patch_width, patch_height = self.patch_size

        return (
            patch_width + conv_size_reduction,
            patch_height + conv_size_reduction
        )

    @property
    def input_dim(self) -> int:
        patch_width, patch_height = self.reduced_patch_size

        return self.get_layer_channels(-1) * patch_width * patch_height

    @property
    def n_patches(self) -> int:
        _, width, height = self.input_shape
        patch_width, patch_height = self.patch_size

        return (width // patch_width) * (height // patch_height)

    def forward(self, images: Tensor) -> Tensor:
        patches = images.unfold(2, self.patch_size[0], self.patch_size[1]) \
            .unfold(3, self.patch_size[0], self.patch_size[1]) \
            .flatten(2, 3)

        if self.conv_layers != 0:
            patches = self.conv(patches)

        patches = patches.transpose(1, 2).flatten(2, 4)

        return self.linear_projection(patches)
