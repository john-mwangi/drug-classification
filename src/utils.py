"""
This module contains utilities for calculating the loss function,
handling categorical variables, neural network model
"""

from numpy.lib.nanfunctions import _remove_nan_1d
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.optim import optimizer

DIMS = 1024


# Defining a class for holding the features and targets
class MoaDataset:
    def __init__(self, dataset, targets) -> None:
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return self.dataset.shape[0]

    # Given a key, returns a row containing the features and targets
    # associated with the key
    def __getitem__(self, item):
        return {
            "X": torch.tensor(data=self.dataset[item, :], dtype=torch.float),
            "y": torch.tensor(data=self.targets[item, :], dtype=torch.float)
        }


# Defining a class for actual training
class Engine:
    def __init__(self, model, optimizer, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train_loss(self, data_loader):

        self.model.train()
        final_loss = 0

        for batch in data_loader:
            self.optimizer.zero_grad()
            inputs = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets=targets, outputs=outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        # Return the average loss
        return final_loss/len(data_loader)

    def eval_loss(self, data_loader):

        self.model.eval()
        final_loss = 0

        for batch in data_loader:
            inputs = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets=targets, outputs=outputs)
            final_loss += loss.item()

        return final_loss / len(data_loader)

    def process_data(df):

        df = df.drop(labels="sig_id", axis=1)
        cat_features = df.select_dtypes(include="object").columns
        ohe = pd.get_dummies(data=df[cat_features])
        df = df.drop(labels=cat_features, axis=1)
        df = df.join(ohe)

        return df


class Model(nn.Model):
    def __init__(self, num_features, num_targets):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(in_features=num_features, out_features=DIMS),
            nn.BatchNorm1d(num_features=DIMS),
            nn.Dropout(p=0.3),
            nn.PReLU(),

            nn.Linear(in_features=DIMS, out_features=DIMS),
            nn.BatchNorm1d(num_features=DIMS),
            nn.Dropout(p=0.3),
            nn.PReLU(),

            nn.Linear(in_features=DIMS, out_features=num_targets)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# https://www.youtube.com/watch?v=VRVit0-0AXE 1:00:00
# Converting the above pytorch model to pytorch-lightning (more elegant)
class MoADataModel(pl.LightningDataModule):
    def __init__(self, hparams, data, targets) -> None:
        super().__init__()
        self.hparams = hparams
        self.data = data
        self.targets = targets

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data, valid_data = train_test_split(
            self.data,
            test_size=0.1,
            random_state=42
            )

        train_targets, valid_targets = train_test_split(
            self.targets,
            test_size=0.1,
            random_state=42
            )

        self.train_dataset = MoaDataset(
            dataset=train_data.iloc[:, 1:].values,
            targets=train_targets.iloc[:, 1:].values
            )

        self.valid_dataset = MoaDataset(
            dataset=valid_data.iloc[:, 1:].values,
            targets=valid_targets.iloc[:, 1:].values
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.dataloader(
            dataset=self.valid_dataset,
            batch_size=DIMS,
            num_workers=0,
            shuffle=False
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.dataloader(
            dataset=self.valid_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=DIMS
        )
        return valid_loader

    def test_dataloader(self):
        return None


class LitMoA(pl.LightningModule):
    def __init__(self, hparams, model) -> None:
        super(LitMoA, self).__init__()
        self.hparams = hparams
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=3,
            threshold=1e-5,
            mode="min",
            verbose=True
        )
        return (
            [optimizer],
            [{
                "scheduler": scheduler,
                "interval": "epoc",
                "monitor": "valid_loss"
            }]
        )
