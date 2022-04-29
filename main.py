from torch import device, cuda
from json import dumps
from argparse import ArgumentParser
from optuna import Trial, create_study, Study

from UnifiedTransformer import UnifiedTransformer
from Trainer import Trainer
from Dataset import MnistDataModule


def objective(trial: Trial) -> float:
    parser = ArgumentParser()
    parser.add_argument('--image-embedding')
    arguments = parser.parse_args()

    MAX_EPOCHS = 50
    LR = 0.01#trial.suggest_float('learning_rate', 1e-4, 1e-2)
    CONV_LAYERS = 0 if arguments.image_embedding != 'convolutional' else trial.suggest_int('conv_layers', 1, 5)
    PATCH_SIZE = (4, 4) if arguments.image_embedding != 'convolutional' else (28, 28)
    DROPOUT = 0.1#trial.suggest_float('dropout', 0.1, 0.4)
    NUM_ENCODER_LAYERS = 1#trial.suggest_int('num_encoder_layers', 1, 6)

    hyperparams = {
        'lr': LR,
        'conv_layers': CONV_LAYERS,
        'dropout': DROPOUT,
        'num_encoder_layers': NUM_ENCODER_LAYERS
    }

    filename = f'saved/{dumps(hyperparams)}.pt'

    print("Hyper parameters: ", hyperparams)

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

    model.to(device("cuda:0" if cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        data_module=data_module,
        n_epochs=MAX_EPOCHS,
        checkpoint_filename=filename
    )

    trainer.fit()

    test_loss = trainer.test()

    return test_loss


def print_best_callback(st: Study, _) -> None:
    print(f"Best value: {st.best_value}, Best params: {st.best_trial.params}")


if __name__ == '__main__':
    study = create_study(direction='minimize')
    study.optimize(objective, n_trials=100, callbacks=[print_best_callback])
