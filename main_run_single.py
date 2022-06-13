from torch import device, cuda, tensor, std_mean
from json import dumps
from argparse import ArgumentParser
from typing import Tuple, Dict
from tabulate import tabulate
import os

from UnifiedTransformer import UnifiedTransformer
from Trainer import Trainer
from Dataset import MnistDataModule


def run(
        hyperparams: Dict,
        patch_size: Tuple[int, int],
        max_epochs: int
) -> float:
    filename = f'saved/{dumps(hyperparams)}.pt'

    print("Hyper parameters: ", hyperparams)

    print("Is Fashion MNIST used? ", hyperparams['dataset'] == 'fashion-mnist')

    data_module = MnistDataModule(fashion_mnist=hyperparams['dataset'] == 'fashion-mnist')

    model = UnifiedTransformer(
        input_shape=(1, 56, 56),
        patch_size=patch_size,
        embed_dim=hyperparams['embedding-dimension'],
        n_heads=hyperparams['num-heads'],
        output_dim=1,
        learning_rate=hyperparams['learning-rate'],
        conv_layers=hyperparams['num-conv-layers'],
        text_length=4,
        dropout=hyperparams['dropout'],
        depth=hyperparams['num-encoder-layers']
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
    parser.add_argument('--dataset')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--embedding-dimension', type=int)
    parser.add_argument('--num-heads', type=int)
    parser.add_argument('--num-conv-layers', type=int)
    parser.add_argument('--num-encoder-layers', type=int)
    arguments = parser.parse_args()

    MAX_EPOCHS = 15
    PATCH_SIZE = (4, 4)
    NUM_RUNS_PER_SETTING = 5

    hyperparams = {
        'dataset': arguments.dataset,
        'learning-rate': arguments.learning_rate,
        'dropout': arguments.dropout,
        'embedding-dimension': arguments.embedding_dimension,
        'num-heads': arguments.num_heads,
        'num-conv-layers': arguments.num_conv_layers,
        'num-encoder-layers': arguments.num_encoder_layers
    }

    LOG_FILENAME = 'logs/' + dumps(hyperparams) + '.log'

    print(f"Starting {NUM_RUNS_PER_SETTING} for hyper parameter setting: ")

    results = [item.item() for item in std_mean(tensor([
        run(hyperparams, PATCH_SIZE, MAX_EPOCHS)
        for _ in range(NUM_RUNS_PER_SETTING)
    ]))]

    # Overwrite log file
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)

    with open(LOG_FILENAME, 'w') as file:
        file.write(tabulate(results))

    print(f"Done with {NUM_RUNS_PER_SETTING} runs!\n\n\n")


if __name__ == '__main__':
    main()
