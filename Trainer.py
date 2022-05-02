from torch import load, no_grad
from torch.utils.data import DataLoader
from typing import Tuple
from alive_progress import alive_bar

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

    def fit(self) -> None:
        train_len = len(self.data_module.train_dataloader())

        for epoch in range(self.n_epochs):
            train_loss = 0.0

            print(f"Epoch {epoch + 1}/{self.n_epochs} | ", end="")

            with alive_bar(train_len) as bar:
                for batch_idx, batch in enumerate(self.data_module.train_dataloader()):
                    loss = self.model.training_step(batch, batch_idx)
                    train_loss += loss.item()

                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()

                    bar()

            train_loss /= train_len

            print(f"Loss: {train_loss:.2f}")

            self.validate()

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

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

    def validate(self) -> None:
        val_loss, val_accuracy = self.test_validate(self.data_module.val_dataloader())

        print("\n\nValidating:")

        self.early_stopping(val_loss, self.model)

        print(f"Validation loss: {val_loss:.2f}")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%\n\n")

    def test(self) -> float:
        self.model.load_state_dict(load(self.checkpoint_filename))

        print("\n\nTesting:")

        test_loss, test_accuracy = self.test_validate(self.data_module.val_dataloader())

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")

        return test_loss, test_accuracy
