from torch import device, cuda, tensor, std_mean
from json import dumps
from argparse import ArgumentParser
from typing import Tuple
import itertools
from random import shuffle
from tabulate import tabulate
import os

from UnifiedTransformer import UnifiedTransformer
from Trainer import Trainer
from Dataset import MnistDataModule


def run(
        lr: float,
        num_conv_layers: int,
        dropout: float,
        num_encoder_layers: int,
        patch_size: Tuple[int, int],
        max_epochs: int,
        fashion_mnist: bool
) -> float:
    hyperparams = {
        'lr': lr,
        'conv_layers': num_conv_layers,
        'dropout': dropout,
        'num_encoder_layers': num_encoder_layers,
        'fashion_mnist': fashion_mnist
    }

    filename = f'saved/{dumps(hyperparams)}.pt'

    print("Hyper parameters: ", hyperparams)

    data_module = MnistDataModule(fashion_mnist=fashion_mnist)

    model = UnifiedTransformer(
        input_shape=(1, 28, 28),
        patch_size=patch_size,
        embed_dim=20,
        n_heads=2,
        output_dim=1,
        learning_rate=lr,
        conv_layers=num_conv_layers,
        text_length=4,
        dropout=dropout,
        depth=num_encoder_layers
    )

    model.to(device("cuda:0" if cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        data_module=data_module,
        n_epochs=max_epochs,
        checkpoint_filename=filename
    )

    trainer.fit()

    _, test_accuracy = trainer.test()

    return test_accuracy


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--image-embedding')
    parser.add_argument('--dataset')
    arguments = parser.parse_args()

    MAX_EPOCHS = 15 if arguments.dataset in ['mnist', 'fashion-mnist'] else 60
    PATCH_SIZE = (4, 4) if arguments.image_embedding != 'convolutional' else (28, 28)
    NUM_RUNS_PER_SETTING = 5

    LOG_FILENAME = 'logs/' + dumps({
        'image-embedding': arguments.image_embedding or 'attention-based',
        'dataset': arguments.dataset or 'mnist'
    }) + '.log'

    # use mnist or fashion mnist as dataset
    FASHION_MNIST = False if arguments.dataset == 'mnist' else 'True'

    # Hyper Parameters for random grid search
    DROPOUT = [0.1, 0.3, 0.4]
    LR = [1e-4, 1e-3, 1e-2]
    NUM_ENCODER_LAYERS = [1, 3, 6]
    CONV_LAYERS = [0] if arguments.image_embedding != 'convolutional' else [1, 3, 5]

    # Random Grid Search
    permutations = list(itertools.product(DROPOUT, LR, NUM_ENCODER_LAYERS, CONV_LAYERS))
    shuffle(permutations)
    results = [[0]] * len(permutations)

    for idx, permutation in enumerate(permutations):
        dropout, lr, num_encoder_layers, num_conv_layers = permutation

        print(f"Starting {NUM_RUNS_PER_SETTING} for hyper parameter setting: ")

        results[idx] = [item.item() for item in std_mean(tensor([
            run(lr, num_conv_layers, dropout, num_encoder_layers, PATCH_SIZE, MAX_EPOCHS, FASHION_MNIST)
            for _ in range(NUM_RUNS_PER_SETTING)
        ]))]

        # Overwrite log file
        if os.path.exists(LOG_FILENAME):
            os.remove(LOG_FILENAME)

        with open(LOG_FILENAME, 'w') as file:
            file.write(tabulate(zip(permutations, results)))

        print(f"Done with {NUM_RUNS_PER_SETTING} runs!\n\n\n")


if __name__ == '__main__':
    main()
