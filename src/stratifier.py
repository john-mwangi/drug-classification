"""
This module splits the data into fold for cross-validation training
"""

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../inputs/train_targets_scored.csv")
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    targets = df.drop("sig_id", axis=1).values

# This is a stratifier designed for multilabel problems
    mlskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (train_, val_) in enumerate(mlskf.split(X=df, y=targets)):
        df.loc[val_, "kfold"] = fold

    df.to_csv("../outputs/train_folds.csv", index=False)
