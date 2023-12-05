import json
import os.path
from os import listdir
from os.path import join
from pathlib import Path
import time
from typing import Callable

import numpy as np
import optuna
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from sklearn.metrics import f1_score

from calculate_map import token_types_to_publaynet_types, get_image_name_image_id, get_one_annotation, map_score
from get_data import get_pdf_name_labels, load_pdf_feature
from train import train
import lightgbm as lgb

TOKEN_TYPE_DATA_PATH = "data/training_data/token_type/train"
SEGMENTATION_DATA_PATH = "data/training_data/segmentation/train"

TOKEN_TYPE_MODEL_PATH = "model/token_type_hyperparams.model"
SEGMENTATION_MODEL_PATH = "model/segmentation_hyperparams.model"

PREDICTIONS_PATH = "data/publaynet/predictions_hyperparams.json"


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


def set_segmentation_predictions(model_configuration: ModelConfiguration, chunk: int):
    pdf_name_labels = get_pdf_name_labels('train', from_document_count=10000 * chunk, to_document_count=10000*(chunk + 1))
    test_pdf_features = [load_pdf_feature('train', x) for x in pdf_name_labels if load_pdf_feature('train', x)]

    print("Predicting token types for", len(test_pdf_features), "pdfs")
    trainer = TokenTypeTrainer(test_pdf_features, ModelConfiguration())
    trainer.set_token_types(TOKEN_TYPE_MODEL_PATH)

    print("Predicting segmentation for", len(test_pdf_features), "pdfs")
    trainer = ParagraphExtractorTrainer(pdfs_features=test_pdf_features, model_configuration=model_configuration)
    segments: list[PdfSegment] = trainer.get_pdf_segments(SEGMENTATION_MODEL_PATH)

    segments = [s for s in segments if s.segment_type in token_types_to_publaynet_types.keys()]

    predictions_coco_format = json.loads(Path(f"data/publaynet/train_chunk_{chunk}.json").read_text())

    image_name_image_id = get_image_name_image_id('train')
    annotations = []
    for i, segment in enumerate(segments):
        annotations.append(get_one_annotation(i, image_name_image_id[segment.pdf_name], segment))

    predictions_coco_format['annotations'] = annotations
    Path(PREDICTIONS_PATH).write_text(json.dumps(predictions_coco_format))


def objective_segmentation(trial: optuna.trial.Trial):
    start = time.time()
    model_configuration = get_model_configuration(trial)

    print(f"Start try of hyperparams at {time.localtime().tm_hour}:{time.localtime().tm_min}")
    print('\t'.join([str(x) for x in model_configuration.dict().keys()]))
    print('\t'.join([str(x) for x in model_configuration.dict().values()]))

    train(model_configuration, SEGMENTATION_DATA_PATH, SEGMENTATION_MODEL_PATH, 3)

    test_chunk = 3
    set_segmentation_predictions(model_configuration, test_chunk)

    coco_score = map_score(truth_path=f"data/publaynet/train_chunk_{test_chunk}.json", prediction_path=PREDICTIONS_PATH)
    save_hyperparameters(model_configuration, coco_score, "results/segmentation_hyperparameters.txt")

    print("finished in", round(time.time() - start, 1), "seconds")

    return coco_score


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
    optuna_automatic_tuning(objective_segmentation)
