from torch import load, no_grad
from torch.utils.data import DataLoader
from typing import Tuple
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from os import getcwd

from Dataset import DataModule
from Utilities import EarlyStopping, ExtendedModule


class Trainer:
    def __init__(
            self,
            model: ExtendedModule,
            data_module: DataModule,
            n_epochs: int = 5,
            checkpoint_filename: str = 'checkpoint.pt'
    ) -> None:
        self.checkpoint_filename = checkpoint_filename
        self.model = model
        self.data_module = data_module

        self.n_epochs = n_epochs

        self.early_stopping = EarlyStopping(patience=3, verbose=True, path=checkpoint_filename)

        font_path = Path(getcwd(), 'assets/fonts/cmuserif.ttf')
        font_manager.FontEntry(
            fname=font_path,
            name='serif'
        )

    def fit(self) -> None:
        train_len = len(self.data_module.train_dataloader())
        train_loss_within_epoch_history = []
        train_loss_history = []
        val_loss_history = []
        acc_history = []  # accuracy starts at 0

        for epoch in range(self.n_epochs):
            train_loss = 0.0

            print(f"Epoch {epoch + 1}/{self.n_epochs}")

            with alive_bar(train_len) as bar:
                for batch_idx, batch in enumerate(self.data_module.train_dataloader()):
                    loss = self.model.training_step(batch, batch_idx)
                    train_loss += loss.item()
                    train_loss_within_epoch_history.append(loss.item())

                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()

                    bar()

            train_loss /= train_len

            train_loss_history.append(train_loss)

            print(f"Loss: {train_loss:.2f}")

            val_loss, val_acc = self.validate()

            val_loss_history.append(val_loss)
            acc_history.append(val_acc)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        val_x = range(train_len, len(train_loss_within_epoch_history) + 1, train_len)
        plt.rcParams.update({
            'font.family': 'serif',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.titlesize': 12
        })
        plt.plot(range(len(train_loss_within_epoch_history)), train_loss_within_epoch_history, label="Training Loss")
        plt.plot(val_x, train_loss_history, label="Training Loss after Epoch")
        plt.plot(val_x, val_loss_history, label="Validation Loss")
        plt.plot(val_x, acc_history, label="Validation Accuracy")
        plt.xlabel("Batch index")
        plt.ylabel("Loss / Accuracy")
        plt.subplots_adjust(bottom=0.25)
        caption = self.checkpoint_filename.split('.pt')[0].split('/')[1]
        caption = caption.split(' ')
        caption.insert(5, '\n')
        caption = ' '.join(caption)
        plt.figtext(0.5, 0.05, r"$\bf{Hyper parameters}$: " + caption, wrap=True, horizontalalignment='center', fontsize=8)
        plt.title("Loss/accuracy throughout training")
        plt.legend(loc='upper right')
        plt.savefig(f"plots/{caption}.png", dpi=300)

        plt.clf()

    def test_validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        test_accuracy = 0.0
        test_loss = 0.0

        with no_grad():
            with alive_bar(len(dataloader)) as bar:
                for batch_idx, batch in enumerate(dataloader):
                    loss, acc = self.model.test_step(batch, batch_idx)
                    test_loss += loss
                    test_accuracy += acc

                    bar()

        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

        return test_loss, test_accuracy

    def validate(self) -> Tuple[float, float]:
        val_loss, val_accuracy = self.test_validate(self.data_module.val_dataloader())

        print("\n\nValidating:")

        self.early_stopping(val_loss, self.model)

        print(f"Validation loss: {val_loss:.2f}")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%\n\n")

        return val_loss, val_accuracy

    def test(self) -> Tuple[float, float]:
        self.model.load_state_dict(load(self.checkpoint_filename))

        print("\n\nTesting:")

        test_loss, test_accuracy = self.test_validate(self.data_module.val_dataloader())

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")

        return test_loss, test_accuracy
