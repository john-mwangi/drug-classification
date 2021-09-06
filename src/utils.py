"""
This module contains utilities for calculating the loss function,
handling categorical variables, and defining the neural network.
"""

import torch
import torch.nn as nn
import pandas as pd

DIMS = 1024


# Defining a class for holding the features and targets
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    # Given a key, returns a row containing the features and targets
    # associated with the key
    def __getitem__(self, item):
        return {
            "x": torch.tensor(data=self.features[item, :], dtype=torch.float),
            "y": torch.tensor(data=self.targets[item, :], dtype=torch.float)
        }


# Defining a class for actual training
class Engine:
    def __init__(self, model, optimizer, device):
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
            inputs = batch["x"].to(self.device)
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
            inputs = batch["x"].to(self.device)
            targets = batch["y"].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets=targets, outputs=outputs)
            final_loss += loss.item()

        return final_loss / len(data_loader)

    def process_data(df):

        sig_id = df[["sig_id"]]
        df = df.drop(labels="sig_id", axis=1)
        cat_features = df.select_dtypes(include="object").columns
        ohe = pd.get_dummies(data=df[cat_features])
        df = df.drop(labels=cat_features, axis=1)
        df = df.join(ohe)
        df = sig_id.join(df)

        return df


class Model(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()

        """
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
        """

        layers = []
        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(in_features=nfeatures,
                                        out_features=hidden_size))
                layers.append(nn.BatchNorm1d(num_features=hidden_size))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_features=hidden_size,
                                        out_features=hidden_size))
                layers.append(nn.BatchNorm1d(num_features=hidden_size))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size,
                                out_features=ntargets))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
