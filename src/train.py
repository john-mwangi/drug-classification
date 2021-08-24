"""
This module is for actual model training.
"""

import pandas as pd
from torch.utils.data import DataLoader

import utils

DEVICE = "cuda"
EPOCHS = 10


def run_training(fold):
    df = pd.read_csv("../inputs/train_features.csv")
    df = utils.process_data(df)
    folds = pd.read_csv("../outputs/train_folds.csv")

    targets = folds.drop(labels=["sig_id", "kfold"], axis=1).columns
    features = df.drop(labels="sig_id", axis=1).columns

    df = pd.merge(left=df, right=folds, how="left", on="sig_id")

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    x_train = train_df[features].to_numpy()
    x_valid = valid_df[features].to_numpy()

    y_train = train_df[targets].to_numpy()
    y_valid = valid_df[targets].to_numpy()

    train_dataset = utils.MoaDataset(dataset=x_train, targets=-y_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        num_workers=2)
