from os.path import join
from pathlib import Path
from time import time

from sklearn.metrics import f1_score, accuracy_score

from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from get_data import load_random_labeled_data

BENCHMARK_MODEL = join(Path(__file__).parent, "model", "benchmark.model")
model_configuration = ModelConfiguration()
model_configuration.resume_training = True
MAX_DOCUMENTS = 15000


def train_for_benchmark():
    Path(BENCHMARK_MODEL).parent.mkdir(exist_ok=True)

    train_pdf_features = load_random_labeled_data(split="train", max_documents=MAX_DOCUMENTS)

    print("documents number:", len(train_pdf_features))
    trainer = TokenTypeTrainer(train_pdf_features, model_configuration)
    labels = [token.token_type.get_index() for token in trainer.loop_tokens()]
    print("nooooow training")
    trainer.train(BENCHMARK_MODEL, labels)


def predict_for_benchmark():
    test_pdf_features = load_random_labeled_data(split="dev")
    print("Prediction PDF number", len(test_pdf_features))
    trainer = TokenTypeTrainer(test_pdf_features, model_configuration)
    truths = [token.token_type.get_index() for token in trainer.loop_tokens()]

    print("predicting")
    trainer.predict(BENCHMARK_MODEL)
    predictions = [token.prediction for token in trainer.loop_tokens()]
    return truths, predictions


def benchmark():
    truths, predictions = predict_for_benchmark()

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)

    Path("LGBM_results.txt").write_text(f"F1 score {f1}%\n" + f"Accuracy score {accuracy}%")
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    print("start")
    start = time()
    # train_for_benchmark()
    benchmark()
    print("finished in", int(time() - start), "seconds")
