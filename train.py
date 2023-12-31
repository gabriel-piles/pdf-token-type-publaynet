from time import time

import lightgbm as lgb
import numpy as np
from os import listdir
from os.path import join
from pathlib import Path
from get_data import balance_data
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration


def train(model_configuration: ModelConfiguration, training_data_path: Path | str, model_path: Path | str, chunks_count: int):

    print(f"Getting model input")

    labels = np.array([])
    x_train = None

    for folder_name in sorted(list(listdir(training_data_path))[:chunks_count-1]):
        start = time()
        print(folder_name)
        if x_train is None:
            x_train = np.load(join(training_data_path, folder_name, "x.npy"))
            labels = np.concatenate((labels, np.load(join(training_data_path, folder_name, "y.npy"))), axis=0)
            x_train, labels = balance_data(x_train, labels)
            print("finished in", round(time() - start, 1), "seconds")
            continue

        x_train_chunk, labels_chunk = balance_data(np.load(join(training_data_path, folder_name, "x.npy")),
                                                   np.load(join(training_data_path, folder_name, "y.npy")))
        x_train = np.concatenate((x_train, x_train_chunk), axis=0)
        labels = np.concatenate((labels, labels_chunk), axis=0)
        print("finished in", round(time() - start, 1), "seconds")

    lgb_train = lgb.Dataset(x_train, labels)
    x_val = np.load(join(training_data_path, sorted(list(listdir(training_data_path)))[chunks_count-1], "x.npy"))
    y_val = np.load(join(training_data_path, sorted(list(listdir(training_data_path)))[chunks_count-1], "y.npy"))
    lgb_val = lgb.Dataset(x_val, y_val)
    print(f"Training with {chunks_count * 10}k samples")
    print(str(x_train.shape))
    gbm = lgb.train(model_configuration.dict(), lgb_train, valid_sets=[lgb_val])

    print(f"Saving")
    Path(model_path).parent.mkdir(exist_ok=True)
    gbm.save_model(model_path, num_iteration=gbm.best_iteration)

