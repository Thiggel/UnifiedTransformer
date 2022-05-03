from torchvision.transforms import ToPILImage
from PIL import ImageFont, ImageDraw, Image
from torch import Tensor, load, tensor, float, cat
from math import sqrt
from typing import List
from random import random

from UnifiedTransformer import UnifiedTransformer
from Dataset import MnistDataModule


def generate_colors(n: int) -> List[List[int]]:
    colors = []

    red = int(256 * random())
    green = int(256 * random())
    blue = int(256 * random())

    step = 256 / n

    for i in range(n):
        red += step
        red %= 256

        green += step
        green %= 256

        blue += step
        blue %= 256

        colors.append([int(red), int(green), int(blue), 255])

    return colors


def create_attention_maps(model: UnifiedTransformer, images: Tensor, text: Tensor) -> None:
    transform = ToPILImage()
    patch_size = (4, 4)

    for layerIdx, layer in enumerate(model.encoder.layers):
        # shape: (32, 2, 201, 201) 'batch_size, attention_head, sequence, sequence'
        # ->     (32, 2, 4, 201)   'batch_size, attention_head, text_sequence, sequence'
        [_, text_attention, _] = layer.attention.attention_buffer.split([196, 4, 1], dim=2)
        # shape: (32, 2, 4, 201) 'batch_size, attention_head, text_sequence, sequence'
        # ->     (32, 2, 4, 196) 'batch_size, attention_head, text_sequence, image_sequence'
        [text_attention, _] = text_attention.split([196, 5], dim=3)

        batch_size, num_heads, num_tokens, num_patches = text_attention.shape
        patches_per_row = int(sqrt(num_patches))
        pixels_per_row = int(patch_size[0] * patches_per_row)

        colors = generate_colors(text.shape[1])

        for datapointIdx, datapoint in enumerate(text_attention):
            for headIdx, head in enumerate(datapoint):

                image = transform(images[datapointIdx])
                text_sequence = text[datapointIdx]

                img = Image.new('RGBA', (512, 512), color=(255, 255, 255))
                draw = ImageDraw.Draw(img, 'RGBA')

                img.paste(image.resize((400, 400)), (56, 20))

                for tokenIdx, token in enumerate(head):

                    attention_map = None
                    currentRow = None

                    for patchIdx, patch_attention in enumerate(token):
                        patch_color = tensor(colors[tokenIdx], dtype=float)
                        patch_color[3] = patch_attention * 255 / 4

                        patch = patch_color.unsqueeze(1).unsqueeze(2).repeat(1, *patch_size)

                        currentRow = patch if currentRow is None else cat((currentRow, patch), dim=1)

                        if int(currentRow.shape[1]) >= pixels_per_row:
                            attention_map = currentRow if attention_map is None else cat((attention_map, currentRow), dim=2)
                            currentRow = None

                    attention_map_image = transform(attention_map.float()).convert('RGBA')

                    temp = Image.new('RGBA', (512, 512), color=(255, 255, 255))
                    temp.paste(attention_map_image.resize((400, 400)), (56, 20))

                    img.paste(Image.alpha_composite(img, temp))

                font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
                for numIdx, num in enumerate(text_sequence.tolist()):
                    draw.text(
                        ((512 - text_sequence.shape[0] * 50) / 2 + 25 + numIdx * 50, 470),
                        str(int(num)),
                        font=font,
                        fill=tuple(colors[numIdx])
                    )

                img.save(
                    f'attention-maps/"layer": {layerIdx}, "datapoint": {datapointIdx}, "head": {headIdx}.png', 'PNG'
                )


if __name__ == '__main__':
    PATCH_SIZE = (4, 4)
    CONV_LAYERS = 0
    LR = 0.01
    DROPOUT = 0.3
    NUM_ENCODER_LAYERS = 3

    data_module = MnistDataModule()

    model = UnifiedTransformer(
        input_shape=(1, 28, 28),
        patch_size=PATCH_SIZE,
        embed_dim=20,
        n_heads=2,
        output_dim=1,
        learning_rate=LR,
        conv_layers=CONV_LAYERS,
        text_length=4,
        dropout=DROPOUT,
        depth=NUM_ENCODER_LAYERS
    )

    model.load_state_dict(load(
        'saved/{"lr": 0.01, "conv_layers": 0,'
        ' "dropout": 0.3, "num_encoder_layers": 3}.pt'
    ))

    test_loader = data_module.test_dataloader()

    batch = next(iter(test_loader))

    [images, numbers], targets = batch

    model(images, numbers)

    create_attention_maps(model, images, numbers)