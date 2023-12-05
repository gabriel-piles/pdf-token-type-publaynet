import os.path
import pickle
from os import listdir
from os.path import join
from pathlib import Path

import numpy as np
import optuna
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import train
from train_token_type import token_type_training_data_path
import lightgbm as lgb


def get_data(data_path: str, label_path: str):
    print("Loading X from: ", data_path)
    print("Loading y from: ", label_path)
    with open(data_path, "rb") as f:
        x_train = pickle.load(f)
    with open(label_path, "rb") as f:
        y_train = pickle.load(f)
    return x_train, np.array(y_train)


def save_hyperparameters(params, roc_auc, content_path):
    if os.path.exists(content_path):
        content = Path(content_path).read_text()
        content += "\n" + '\t'.join([str(roc_auc)] + list(params.dict().values()))
        Path(content_path).write_text(content)
    else:
        Path(content_path).write_text('\t'.join(["Roc Auc"] + list(params.dict().keys())) + "\n" +
                                      '\t'.join([str(roc_auc)] + list(params.dict().values())))


def objective_token_type(trial: optuna.trial.Trial):
    params = ModelConfiguration()
    params.deterministic = False
    params.num_leaves = trial.suggest_int("num_leaves", 100, 500)
    params.bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)
    params.bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    params.feature_fraction = trial.suggest_float("feature_fraction", 0.1, 1.0)
    params.lambda_l1 = trial.suggest_float("lambda_l1", 1e-08, 10.0, log=True)
    params.lambda_l2 = trial.suggest_float("lambda_l2", 1e-08, 10.0, log=True)
    params.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 100)
    train(params, TOKEN_TYPE_DATA_PATH, TOKEN_TYPE_MODEL_PATH, 7)
    model = lgb.Booster(model_file=TOKEN_TYPE_MODEL_PATH)
    x_test = np.load(join(TOKEN_TYPE_DATA_PATH, sorted(list(listdir(TOKEN_TYPE_DATA_PATH)))[7], "x.npy"))
    y_test = np.load(join(TOKEN_TYPE_DATA_PATH, sorted(list(listdir(TOKEN_TYPE_DATA_PATH)))[7], "y.npy"))
    y_pred_scores = model.predict(x_test, num_iteration=model.best_iteration)
    roc_auc = roc_auc_score(y_test, y_pred_scores[:, 1], multi_class="ovr")
    save_hyperparameters(params, roc_auc, "results/token_type_hyperparameters.txt")

    return roc_auc


def objective_segmentation(trial: optuna.trial.Trial):
    params = ModelConfiguration()
    params.context_size = 1
    params.num_class = 2
    params.deterministic = False
    params.num_leaves = trial.suggest_int("num_leaves", 100, 500)
    params.bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)
    params.bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    params.feature_fraction = trial.suggest_float("feature_fraction", 0.1, 1.0)
    params.lambda_l1 = trial.suggest_float("lambda_l1", 1e-08, 10.0, log=True)
    params.lambda_l2 = trial.suggest_float("lambda_l2", 1e-08, 10.0, log=True)
    params.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 100)
    train(params, SEGMENTATION_DATA_PATH, SEGMENTATION_MODEL_PATH, 7)
    model = lgb.Booster(model_file=SEGMENTATION_MODEL_PATH)
    x_test = np.load(join(SEGMENTATION_DATA_PATH, sorted(list(listdir(SEGMENTATION_DATA_PATH)))[7], "x.npy"))
    y_test = np.load(join(SEGMENTATION_DATA_PATH, sorted(list(listdir(SEGMENTATION_DATA_PATH)))[7], "y.npy"))
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    roc_auc = roc_auc_score(y_test, y_pred, multi_class="ovr")
    save_hyperparameters(params, roc_auc, "results/segmentation_hyperparameters.txt")

    return roc_auc


def optuna_automatic_tuning(task: str):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial: ")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    result_string: str = ""
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")
        result_string += f'"{key}": {value},\n'
    result_string += f"-> Best trial value: {str(trial.value)}\n"

    result_string += "\n\n\n"

    with open(f"src/tuned_parameters/{task}.txt", "a") as result_file:
        result_file.write(result_string)


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