from os import listdir
from os.path import join
from pathlib import Path
from time import time

import numpy as np
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from get_data import load_labeled_data
import lightgbm as lgb


TOKEN_TYPE_MODEL_PATH = join(Path(__file__).parent, "model", "token_type.model")

model_configuration = ModelConfiguration()
model_configuration.context_size = 4
model_configuration.num_boost_round = 200
model_configuration.num_leaves = 100

MAX_DOCUMENTS = 10000
token_type_training_data_path = join("data", "training_data", "token_type", "train")


def train_token_type():

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
    gbm = lgb.train(model_configuration.dict(), lgb_train)

    print(f"Saving")
    Path(TOKEN_TYPE_MODEL_PATH).parent.mkdir(exist_ok=True)
    gbm.save_model(TOKEN_TYPE_MODEL_PATH, num_iteration=gbm.best_iteration)


def cache_token_type_training_data():
    for i in range(8):
        start_cache = time()
        print(f"Caching chunk {i}")
        train_pdf_features = load_labeled_data(split="train", from_document_count=MAX_DOCUMENTS * i,
                                               to_document_count=MAX_DOCUMENTS * (i + 1))

        if not train_pdf_features:
            continue

        trainer = TokenTypeTrainer(train_pdf_features, model_configuration)
        labels = [token.token_type.get_index() for token in trainer.loop_tokens()]
        trainer.save_training_data(join(token_type_training_data_path, f"chunk_{i}"), labels)
        print("finished in", round(time() - start_cache, 1), "seconds\n")


def cache_validation_data():
    start_cache = time()
    print("loading data")
    val_pdf_features = load_labeled_data(split="dev", from_document_count=0, to_document_count=999999999)
    print("finished in", round(time() - start_cache, 1), "seconds")

    start_cache = time()

    print("saving data")
    trainer = TokenTypeTrainer(val_pdf_features, model_configuration)
    labels = [token.token_type.get_index() for token in trainer.loop_tokens()]
    trainer.save_training_data(join("data", "training_data", "token_type", "val", "chunk_0"), labels)
    print("finished in", round(time() - start_cache, 1), "seconds")


if __name__ == "__main__":
    start = time()
    print("train_token_type")

    cache_token_type_training_data()
    print("finished in", int(time() - start), "seconds")

