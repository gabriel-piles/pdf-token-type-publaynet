from os import listdir
from os.path import join
from pathlib import Path

import numpy as np

from train_token_type import token_type_training_data_path
import lightgbm as lgb


def optimize():
    pass


def train_token_type(configuration, model_path):

    print(f"Getting model input")

    labels = np.array([])
    x_train = None

    for folder_name in list(listdir(token_type_training_data_path))[:2]:
        if x_train is None:
            x_train = np.load(join(token_type_training_data_path, folder_name, "x.npy"))
        else:
            x_train = np.concatenate((x_train, np.load(join(token_type_training_data_path, folder_name, "x.npy"))), axis=0)

        labels = np.concatenate((labels, np.load(join(token_type_training_data_path, folder_name, "y.npy"))), axis=0)

    lgb_train = lgb.Dataset(x_train, labels)
    print(f"Training")
    print(str(x_train.shape))
    gbm = lgb.train(configuration.dict(), lgb_train)

    print(f"Saving")
    Path(model_path).parent.mkdir(exist_ok=True)
    gbm.save_model(model_path, num_iteration=gbm.best_iteration)


if __name__ == '__main__':
    optimize()