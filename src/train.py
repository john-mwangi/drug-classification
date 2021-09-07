"""
This module is for training the actual neural network and hyper-parameter
tuning.
"""

import pandas as pd
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader

import utils

DEVICE = "cpu"
EPOCHS = 10


def run_training(fold, params, save_model=False):
    df = pd.read_csv("../inputs/train_features.csv")
    df = utils.Engine.process_data(df=df)

    targets_df = pd.read_csv("../outputs/train_targets_folds.csv")

    features_columns = df.drop(labels="sig_id", axis=1).columns
    target_columns = targets_df.drop(["sig_id", "kfold"], axis=1).columns

    df = pd.merge(left=df, right=targets_df, how="left", on="sig_id")

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[features_columns].to_numpy()
    xvalid = valid_df[features_columns].to_numpy()

    ytrain = train_df[target_columns].to_numpy()
    yvalid = valid_df[target_columns].to_numpy()

    train_dataset = utils.MoADataset(features=xtrain, targets=-ytrain)
    valid_dataset = utils.MoADataset(features=xvalid, targets=yvalid)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        num_workers=2,
        shuffle=True
        )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1024,
        num_workers=2
    )

    model = utils.Model(
        nfeatures=xtrain.shape[1],
        ntargets=ytrain.shape[1],
        nlayers=params["num_layers"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"]
    )

    model.to(device=DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        patience=3,
        threshold=1e-5,
        verbose=True
    )

    eng = utils.Engine(model=model, optimizer=optimizer, device=DEVICE)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        train_loss = eng.train_loss(data_loader=train_loader)
        valid_loss = eng.eval_loss(data_loader=valid_loader)
        scheduler.step(valid_loss)
        print(f"{fold}, {epoch}, {train_loss}, {valid_loss}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(obj=model.state_dict(), f=f"model_{fold}.bin")
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break

    return best_loss


def objective(trial):
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 7),
        "hidden_size": trial.suggest_int("hidden_size", 16, 2048),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    }

    all_losses = []
    for f_ in range(5):
        temp_loss = run_training(fold=f_, params=params, save_model=False)
        all_losses.append(temp_loss)

    return np.mean(all_losses)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(func=objective, n_trials=20)

    print("Best trial:")
    trial_ = study.best_trial

    print(trial_.values)
    print(trial_.params)

    scores = 0
    for j in range(5):
        scr = run_training(fold=j, params=trial_.params, save_model=True)
        scores += scr

    print(f"CV Score of best params: {scores/5}")
