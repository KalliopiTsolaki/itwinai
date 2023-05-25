"""
Custom Pytorch Lightning models for MNIST dataset.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision import transforms

from itwinai.plmodels.base import ItwinaiBasePlModule


class LitMNIST_Custom(ItwinaiBasePlModule):
    """
    Simple PL model for MNIST.
    Adapted from
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
    """

    def __init__(
        self,
        hidden_size: int = 64,
    ):
        super().__init__()

        # Automatically save constructor args as hyperparameters
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.hidden_size = hidden_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log metrics with autolog
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
