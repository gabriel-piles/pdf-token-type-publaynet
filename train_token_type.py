from os.path import join
from pathlib import Path
from time import time

from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from get_data import load_labeled_data
from train import train

TOKEN_TYPE_MODEL_PATH = join(Path(__file__).parent, "model", "token_type.model")

MAX_DOCUMENTS = 10000
token_type_training_data_path = join("data", "training_data", "token_type", "train")


def train_token_type():
    configuration_dict = dict()
    configuration_dict["context_size"] = 4
    configuration_dict["num_boost_round"] = 652
    configuration_dict["num_leaves"] = 427
    configuration_dict["bagging_fraction"] = 0.538576362137786
    configuration_dict["lambda_l1"] = 0.0005757631630210714
    configuration_dict["lambda_l2"] = 0.014421883568195633
    configuration_dict["feature_fraction"] = 0.5093595305684682
    configuration_dict["bagging_freq"] = 4
    configuration_dict["min_data_in_leaf"] = 91
    configuration_dict["feature_pre_filter"] = False
    configuration_dict["boosting_type"] = "gbdt"
    configuration_dict["objective"] = "multiclass"
    configuration_dict["metric"] = "multi_logloss"
    configuration_dict["learning_rate"] = 0.1
    configuration_dict["seed"] = 22
    configuration_dict["num_class"] = 13
    configuration_dict["verbose"] = -1
    configuration_dict["deterministic"] = False
    configuration_dict["resume_training"] = False

    model_configuration = ModelConfiguration(**configuration_dict)

    train(model_configuration=model_configuration,
          training_data_path="data/training_data/token_type/train",
          model_path=TOKEN_TYPE_MODEL_PATH,
          chunks_count=33)


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

    train_token_type()
    print("finished in", int(time() - start), "seconds")

