from os import listdir
from os.path import join
from pathlib import Path
from time import time

import numpy as np
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

from calculate_map import map_score
from get_data import get_segmentation_labeled_data, balance_data

from hyperparams import set_segmentation_predictions, PREDICTIONS_PATH
from train import train

MAX_DOCUMENTS = 10000

SEGMENTATION_MODEL_PATH = join(Path(__file__).parent, "model", "4_jan_2024_segmentation_model.model")

configuration_dict = dict()
configuration_dict["context_size"] = 1
configuration_dict["num_boost_round"] = 331
configuration_dict["num_leaves"] = 326
configuration_dict["bagging_fraction"] = 0.8741546573792001
configuration_dict["lambda_l1"] = 3.741871910299135e-07
configuration_dict["lambda_l2"] = 3.394743918196975e-07
configuration_dict["feature_fraction"] = 0.17453493249431365
configuration_dict["bagging_freq"] = 9
configuration_dict["min_data_in_leaf"] = 35
configuration_dict["feature_pre_filter"] = False
configuration_dict["boosting_type"] = "gbdt"
configuration_dict["objective"] = "multiclass"
configuration_dict["metric"] = "multi_logloss"
configuration_dict["learning_rate"] = 0.1
configuration_dict["seed"] = 22
configuration_dict["num_class"] = 2
configuration_dict["verbose"] = -1
configuration_dict["deterministic"] = False
configuration_dict["resume_training"] = False

model_configuration = ModelConfiguration(**configuration_dict)


def loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list: list[PdfParagraphTokens]):
    for pdf_paragraph_tokens in pdf_paragraph_tokens_list:
        for page in pdf_paragraph_tokens.pdf_features.pages:
            if not page.tokens:
                continue
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                yield pdf_paragraph_tokens, token, next_token
            yield pdf_paragraph_tokens, page.tokens[-1], page.tokens[-1]


def train_segmentation(chunks_count = 33):
    train(model_configuration=model_configuration,
          training_data_path="data/training_data/segmentation/train",
          model_path=SEGMENTATION_MODEL_PATH,
          chunks_count=chunks_count)


def cache_training_data(split: str, chunks_list: list[int]):
    for i in chunks_list:
        start_cache = time()
        print(f"Caching chunk {i}")
        pdf_paragraph_tokens_list = get_segmentation_labeled_data(split="train" if split == "train" else "dev", from_document_count=MAX_DOCUMENTS * i,
                                                                  to_document_count=MAX_DOCUMENTS * (i + 1))

        if not pdf_paragraph_tokens_list:
            continue

        pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]

        trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=model_configuration)
        labels = []
        for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
            labels.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))

        chunk_folder = "train" if split == "train" else "val"
        trainer.save_training_data(join("data", "training_data", "segmentation", chunk_folder, f"chunk_{i}"), labels)
        print("finished in", round(time() - start_cache, 1), "seconds\n")


def evaluate_results(test_chunk= 33):
    set_segmentation_predictions(model_configuration=model_configuration,
                                 segmentation_model_path=SEGMENTATION_MODEL_PATH,
                                 chunk=test_chunk)

    coco_score = map_score(truth_path=f"data/publaynet/train_chunk_{test_chunk}.json", prediction_path=PREDICTIONS_PATH)
    return coco_score


def load_chunks():
    training_data_path = "data/training_data/token_type/train"
    # shapes = list()
    # for chunk in range(7):
    #     data = np.load(f"{training_data_path}/chunk_{chunk}/x.npy")
    #     shapes.append(data.shape[0])
    #     print(data.shape)

    x_train = None
    labels = np.array([])
    for folder_name in sorted(list(listdir(training_data_path))[:4]):

        if x_train is None:
            x_train = np.load(join(training_data_path, folder_name, "x.npy"))
            labels = np.concatenate((labels, np.load(join(training_data_path, folder_name, "y.npy"))), axis=0)
            x_train, labels = balance_data(x_train, labels)
            print("finished in", round(time() - start, 1), "seconds")
            continue

        x_train_chunk, labels_chunk = balance_data(np.load(join(training_data_path, folder_name, "x.npy")),
                                                   np.load(join(training_data_path, folder_name, "y.npy")))

        labels = np.concatenate((labels, labels_chunk), axis=0)

        start_loop = time()
        x_train = np.concatenate((x_train, x_train_chunk), axis=0)

        # rows = x_train_chunk.shape[0]
        # old_rows = x_train.shape[0]
        #
        # X = np.zeros((rows + old_rows, 400))
        # X[:old_rows] = x_train
        # X[old_rows:] = x_train_chunk
        # x_train = X
        print("finished in", round(time() - start_loop, 1), "seconds")


def performance_over_time():
    for i in range(20):
        train_segmentation(i + 2)
        map = evaluate_results(32)
        results_path = Path("results/segmentation_performance_over_size.txt")
        results = results_path.read_text()
        results_path.write_text(f"{results}\n{i}\t{map}")


if __name__ == '__main__':
    print("start")
    start = time()
    cache_training_data("train", list(range(18)))
    print("finished in", int(time() - start), "seconds")
