import optuna
import torch
from torch import nn
import pandas as pd
import numpy as np

from functools import partial
import utils

DEVICE = "cuda"
EPOCHS = 4


class ModelX(nn.Module):
    def __init__(self, num_features, num_targets, num_layers, hidden_size,
                 dropout):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(in_features=num_features,
                                        out_features=hidden_size))
                layers.append(nn.BatchNorm1d(num_features=hidden_size))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.PReLU())
            else:
                layers.append(nn.Linear(in_features=hidden_size,
                                        out_features=hidden_size))
                layers.append(nn.BatchNorm1d(num_features=hidden_size))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.PReLU())
        layers.append(nn.Linear(in_features=hidden_size,
                                out_features=num_targets))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def run_training(fold, params, save_model=False):
        df = pd.read_csv("../inputs/train_features.csv")
        df = utils.Engine.process_data(df=df)
        folds = pd.read_csv("../outputs/train_folds.csv")

        non_scored_df = pd.read_csv("../inputs/train_targets_nonscored.csv")
        non_scored_targets = non_scored_df.drop(labels="sig_id", axis=1)
        non_scored_targets = non_scored_targets.to_numpy().sum(axis=1)
        non_scored_df.loc[:, "nscr"] = non_scored_targets

        drop_cols = [c for c in non_scored_df.columns if c not in ("nscr", "sig_id")]
        non_scored_df = non_scored_df.drop(labels=drop_cols, axis=1)
        folds = folds.merge(right=non_scored_df, on="sig_id", how="left")

        targets = folds.drop(labels=["sig_id", "kfold"], axis=1).columns
        features = df.drop("sig_id", axis=1).columns

        df = df.merge(folds, on="sig_id", how="left")

        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        x_train = train_df[features].to_numpy()
        x_valid = valid_df[features].to_numpy()

        y_train = train_df[targets].to_numpy()
        y_valid = valid_df[targets].to_numpy()

        train_dataset = utils.MoADataset(dataset=x_train, targets=y_train)
        train_loader = torch.utils.data.dataloader(
            train_dataset, batch_size=1024, num_workers=2
        )

        valid_dataset = utils.MoADataset(dataset=x_valid, targets=y_valid)
        valid_loader = torch.utils.data.dataloader(
            valid_dataset, batch_size=1024, num_workers=2
        )

        model = ModelX(
            num_features=x_train.shape[1],
            num_targets=y_train.shape[1],
            num_layers=params["num_layers"],
            hidden_size=params["hidden_size"],
            dropout=params["dropout"]
        )

        model.to(DEVICE)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=params["learning_rate"]
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=3,
            threshold=1e-5,
            mode="min",
            verbose=True
        )

        eng = utils.Engine(
            model=model,
            optimizer=optimizer,
            device=DEVICE
        )

        best_loss = np.inf
        early_stopping = 10
        early_stopping_counter = 0
        for epoch in range(EPOCHS):
            train_loss = eng.train(train_loader)
            valid_loss = eng.valid(valid_loader)
            scheduler.step(valid_loss)
            print(
                f"fold={fold}, \
                epoch={epoch}, \
                train_loss={train_loss}, \
                valid_loss={valid_loss}"
                )

            if valid_loss < best_loss:
                best_loss = valid_loss
                if save_model:
                    torch.save(
                        obj=model.state_dict(),
                        f=f"model_fold{fold}.bin")
            else:
                early_stopping_counter += 1

            if early_stopping_counter > early_stopping:
                break

        print(f"fold={fold}, best validation loss={best_loss}")
        return best_loss

    def objective(self, trial):
        params = {
            "num_layers": trial.suggest_int("num_layers", 1, 7),
            "hidden_size": trial.suggest_int("hidden_size", 16, 2048),
            "dropout": trial.suggest_uniform("dropout", 0.1, 0.8),
            "learning_rate": trial.suggest_loguniform("learning_rate",
                                                      1e-6, 1e-3)
        }

        all_loss = []
        for fold_ in range(5):
            temp_loss = self.run_training(
                fold_,
                params=params,
                save_model=False)
            all_loss.append(temp_loss)
        return np.mean(all_loss)

    if __name__ == "__main__":
        partial_obj = partial(objective)
        study = optuna.create_study(direction="minimize")
        study.optimize(func=partial_obj, n_trials=5)

        print("Best trial:")
        trail_ = study.best_trial
        print(f"Value: {trail_.value}")

        print("Params:")
        best_params = trail_.best_params
        print(best_params)

        scores = 0
        for j in range(5):
            score = run_training(fold=j, params=best_params, save_model=True)
            scores += score

        print(f"OOF Score: {scores/5}")
