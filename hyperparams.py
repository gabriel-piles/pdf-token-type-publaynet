import os.path
from os import listdir
from os.path import join
from pathlib import Path
import time
from typing import Callable

import numpy as np
import optuna
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from sklearn.metrics import f1_score

from train import train
import lightgbm as lgb

TOKEN_TYPE_DATA_PATH = "data/training_data/token_type/train"
SEGMENTATION_DATA_PATH = "data/training_data/segmentation/train"

TOKEN_TYPE_MODEL_PATH = "model/token_type_hyperparams.model"
SEGMENTATION_MODEL_PATH = "model/segmentation_hyperparams.model"


def save_hyperparameters(model_configuration, f1, content_path):
    string_f1 = f"{f1:.3}"
    if os.path.exists(content_path):
        content = Path(content_path).read_text()
        content += "\n" + '\t'.join([string_f1] + [str(x) for x in model_configuration.dict().values()])
        Path(content_path).write_text(content)
    else:
        Path(content_path).write_text('\t'.join(["f1"] + [str(x) for x in model_configuration.dict().keys()]) + "\n" +
                                      '\t'.join([string_f1] + [str(x) for x in model_configuration.dict().values()]))


def objective_token_type(trial: optuna.trial.Trial):
    start = time.time()
    model_configuration = get_model_configuration(trial)

    print(f"Start try of hyperparams at {time.localtime().tm_hour}:{time.localtime().tm_min}")
    print('\t'.join([str(x) for x in model_configuration.dict().keys()]))
    print('\t'.join([str(x) for x in model_configuration.dict().values()]))

    train(model_configuration, TOKEN_TYPE_DATA_PATH, TOKEN_TYPE_MODEL_PATH, 7)
    model = lgb.Booster(model_file=TOKEN_TYPE_MODEL_PATH)
    chunks_paths = sorted(list(listdir(TOKEN_TYPE_DATA_PATH)))

    x_test = np.load(join(TOKEN_TYPE_DATA_PATH, chunks_paths[7], "x.npy"))
    y_test = np.load(join(TOKEN_TYPE_DATA_PATH, chunks_paths[7], "y.npy"))

    y_pred_scores = model.predict(x_test, num_iteration=model.best_iteration)
    y_prediction_categories = [np.argmax(prediction_scores) for prediction_scores in y_pred_scores]

    f1 = f1_score(y_test, y_prediction_categories, average="macro")
    save_hyperparameters(model_configuration, f1, "results/token_type_hyperparameters.txt")

    print("finished in", round(time.time() - start, 1), "seconds")

    return f1


def get_model_configuration(trial, is_segmentation: bool = False):
    model_configuration = ModelConfiguration()

    if is_segmentation:
        model_configuration.context_size = 1
        model_configuration.num_class = 2

    model_configuration.deterministic = False
    model_configuration.num_boost_round = trial.suggest_int("num_boost_round", 300, 700)
    model_configuration.num_leaves = trial.suggest_int("num_leaves", 100, 500)
    model_configuration.bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)
    model_configuration.bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    model_configuration.feature_fraction = trial.suggest_float("feature_fraction", 0.1, 1.0)
    model_configuration.lambda_l1 = trial.suggest_float("lambda_l1", 1e-08, 10.0, log=True)
    model_configuration.lambda_l2 = trial.suggest_float("lambda_l2", 1e-08, 10.0, log=True)
    model_configuration.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 100)
    return model_configuration


def optuna_automatic_tuning(objective: Callable):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, gc_after_trial=True)


if __name__ == '__main__':
    optuna_automatic_tuning(objective_token_type)
