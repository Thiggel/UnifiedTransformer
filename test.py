from torchvision.transforms import ToPILImage
from PIL import ImageFont, ImageDraw, Image
from torch import Tensor, load, tensor, float, cat
from math import sqrt
from typing import List, Tuple
from random import random
from json import dumps

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


def generate_attention_map(token, tokenIdx, patch_size, pixels_per_row, colors):
    transform = ToPILImage()

    attention_map = None
    currentRow = None

    for patchIdx, patch_attention in enumerate(token):
        # recreate patches of 4x4 (or nxn) pixels with transparency corresponding to
        # attention value
        patch_color = tensor(colors[tokenIdx], dtype=float)
        patch_color[3] = patch_attention * 255 / 4

        patch = patch_color.unsqueeze(1).unsqueeze(2).repeat(1, *patch_size)

        # we concatenate them in rows and later rows into images
        currentRow = patch if currentRow is None else cat((currentRow, patch), dim=1)

        # whenever one row is full, we append it to the map and truncate it
        if int(currentRow.shape[1]) >= pixels_per_row:
            attention_map = currentRow if attention_map is None else cat((attention_map, currentRow), dim=2)
            currentRow = None

    # return the attention map image
    return transform(attention_map.float()).convert('RGBA')


def paste_attention_maps_in_image(img, head, patch_size, pixels_per_row, colors):
    # we go through each text token and color it + the portions of the image it attends to
    for tokenIdx, token in enumerate(head):
        attention_map_image = generate_attention_map(token, tokenIdx, patch_size, pixels_per_row, colors)

        temp = Image.new('RGBA', (512, 512), color=(255, 255, 255))
        temp.paste(attention_map_image.resize((400, 400)), (56, 20))

        img.paste(Image.alpha_composite(img, temp))


def paste_text_in_image(draw, text_sequence, colors):
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
    for numIdx, num in enumerate(text_sequence.tolist()):
        draw.text(
            ((512 - text_sequence.shape[0] * 50) / 2 + 25 + numIdx * 50, 470),
            str(int(num)),
            font=font,
            fill=tuple(colors[numIdx])
        )


def generate_full_attention_map_image(images, datapointIdx, text, head, patch_size, pixels_per_row, colors, layerIdx, headIdx):
    transform = ToPILImage()

    # set up the image
    image = transform(images[datapointIdx])
    text_sequence = text[datapointIdx]

    img = Image.new('RGBA', (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    img.paste(image.resize((400, 400)), (56, 20))

    paste_attention_maps_in_image(img, head, patch_size, pixels_per_row, colors)

    paste_text_in_image(draw, text_sequence, colors)

    img.save(
        f'attention-maps/"datapoint": {datapointIdx}, "layer": {layerIdx}, "head": {headIdx}.png', 'PNG'
    )


def n_patches(input_shape, patch_size) -> int:
    _, width, height = input_shape
    patch_width, patch_height = patch_size

    return (width // patch_width) * (height // patch_height)

def create_attention_maps(model: UnifiedTransformer, images: Tensor, text: Tensor, patch_size: Tuple[int, int]) -> None:
    num_patches = n_patches((1, 56, 56), patch_size)

    # after processing the input, the buffers in the network contain the attention
    # for the entire batch, so we first loop through all layers for extracting the
    # attention data
    for layerIdx, layer in enumerate(model.encoder.layers):
        # shape: (32, 2, 201, 201) 'batch_size, attention_head, sequence, sequence'
        # ->     (32, 2, 4, 201)   'batch_size, attention_head, text_sequence, sequence'
        [_, text_attention, _] = layer.attention.attention_buffer.split([num_patches, 4, 1], dim=2)
        # shape: (32, 2, 4, 201) 'batch_size, attention_head, text_sequence, sequence'
        # ->     (32, 2, 4, 196) 'batch_size, attention_head, text_sequence, image_sequence'
        [text_attention, _] = text_attention.split([num_patches, 5], dim=3)

        batch_size, num_heads, num_tokens, num_patches = text_attention.shape
        patches_per_row = int(sqrt(num_patches))
        pixels_per_row = int(patch_size[0] * patches_per_row)

        colors = generate_colors(text.shape[1])

        # then we loop through all data points inside the batch
        for datapointIdx, datapoint in enumerate(text_attention):
            # each attention layer has multiple heads
            for headIdx, head in enumerate(datapoint):
                generate_full_attention_map_image(
                    images, datapointIdx, text, head, patch_size, pixels_per_row, colors, layerIdx, headIdx
                )


def main() -> None:
    CONV_LAYERS = 0
    PATCH_SIZE = (4, 4) if CONV_LAYERS == 0 else (28, 28)
    LR = 0.01
    DROPOUT = 0.1
    NUM_ENCODER_LAYERS = 3
    FASHION_MNIST = False

    data_module = MnistDataModule(fashion_mnist=FASHION_MNIST == "True")

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

    hyperparams = {
        'lr': LR,
        'conv_layers': CONV_LAYERS,
        'dropout': DROPOUT,
        'num_encoder_layers': NUM_ENCODER_LAYERS,
        'fashion_mnist': FASHION_MNIST
    }

    filename = f'saved/{dumps(hyperparams)}.pt'

    model.load_state_dict(load(filename))

    test_loader = data_module.test_dataloader()

    batch = next(iter(test_loader))

    [images, numbers], targets = batch

    model(images, numbers)

    create_attention_maps(model, images, numbers, PATCH_SIZE)


if __name__ == '__main__':
    main()
