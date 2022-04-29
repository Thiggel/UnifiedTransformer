from torch import rand, stack, vstack, ones, cat, Tensor, device, cuda
from torch.nn import Sequential, \
    Linear, \
    Softmax, \
    Sigmoid, \
    Embedding, \
    Parameter, \
    BCELoss, \
    CrossEntropyLoss
from torch.optim import Adam
from typing import Tuple
from torchmetrics import Accuracy
from torchvision.transforms import ToPILImage
from PIL import ImageFont, ImageDraw, Image

from Utilities import ExtendedModule
from Transformer import PatchEmbedding, PositionalEncoding, Encoder


class UnifiedTransformer(ExtendedModule):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (1, 28, 28),
            patch_size: Tuple[int, int] = (4, 4),
            conv_layers: int = 0,
            text_length: int = 1,
            embed_dim: int = 8,
            n_heads: int = 2,
            vocab_size: int = 10,
            output_dim: int = 1,
            learning_rate: float = 1e-3,
            depth: int = 1,
            dropout: float = 0.1,
            show_attention_maps: bool = False
    ):
        super(UnifiedTransformer, self).__init__()

        self.show_attention_maps = show_attention_maps

        self.output_dim = output_dim

        self.patch_embedding = PatchEmbedding(input_shape, patch_size, embed_dim, conv_layers)

        self.sequence_length = text_length * self.patch_embedding.n_patches + text_length + 1

        self.embedding = Embedding(vocab_size, embed_dim)

        self.class_token = Parameter(rand(1, embed_dim))

        self.positional_encoding = PositionalEncoding(embed_dim, self.sequence_length)
        self.encoder = Encoder(self.sequence_length, input_shape, n_heads, embed_dim, depth, dropout)

        self.MLP = Sequential(
            Linear(embed_dim, output_dim),
            Sigmoid() if output_dim == 1 else Softmax()
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = BCELoss() if output_dim == 1 else CrossEntropyLoss()

        self.accuracy = Accuracy()

        self.dev = device("cuda:0" if cuda.is_available() else "cpu")

    @staticmethod
    def create_pad_mask(tensor: Tensor, prepended_trues_for_patches: Tuple[int, int]) -> Tensor:
        tensor = cat((ones(prepended_trues_for_patches[0:2]), tensor), dim=1)

        return (tensor == 0).byte()

    def create_attention_maps(self, images, text):
        if not self.show_attention_maps:
            return

        transform = ToPILImage()

        image = transform(images[0])
        text = text[0]

        img = Image.new('RGB', (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        img.paste(image.resize((400, 400)), (56, 20))

        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
        draw.text((150, 470), '          '.join([str(int(num)) for num in text.tolist()]), font=font, fill=(0, 0, 0))

        img.save('attention-map.jpg')

        exit()

        for layer in self.encoder.layers:
            print(layer.attention.attention_buffer.shape)

    def forward(self, images, text):
        images_embedded = self.patch_embedding(images)

        text_embedded = self.embedding(text.long())

        tokens = cat((images_embedded, text_embedded), dim=1)

        tokens = stack([
            vstack((self.class_token, tokens[i])) for i in range(len(tokens))
        ])

        positionally_encoded = self.positional_encoding(tokens)

        encoded = self.encoder(positionally_encoded)

        self.create_attention_maps(images, text)

        final_class_tokens = encoded[:, 0]

        output = self.MLP(final_class_tokens)

        if self.output_dim == 1:
            output = output.flatten()

        return output

    def step(self, batch: Tensor, _: int) -> Tuple[Tensor, Tensor]:
        (images, numbers), targets = batch

        images, numbers, targets = images.to(self.dev), numbers.to(self.dev), targets.to(self.dev)

        y_hat = self(images, numbers)
        loss = self.loss_fn(y_hat, targets.float())
        acc = self.accuracy(y_hat, targets.long())

        return loss, acc

    def training_step(self, batch: Tensor, _: int) -> Tensor:
        loss, _ = self.step(batch, _)

        return loss

    def validation_step(self, batch: Tensor, _: int) -> Tuple[Tensor, Tensor]:
        return self.step(batch, _)

    def test_step(self, batch: Tensor, _: int) -> Tuple[Tensor, Tensor]:
        return self.step(batch, _)
