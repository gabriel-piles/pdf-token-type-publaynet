from pathlib import Path
from time import time
from sklearn.metrics import f1_score, accuracy_score

from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer

from get_data import load_labeled_data
from train_token_type import TOKEN_TYPE_MODEL_PATH

model_configuration = ModelConfiguration()
model_configuration.resume_training = True
MAX_DOCUMENTS = 15000


def predict_token_type_for_benchmark():
    test_pdf_features = load_labeled_data(split="dev")
    print("Prediction PDF number", len(test_pdf_features))
    trainer = TokenTypeTrainer(test_pdf_features, model_configuration)
    truths = [token.token_type.get_index() for token in trainer.loop_tokens()]

    print("predicting")
    trainer.predict(TOKEN_TYPE_MODEL_PATH)
    predictions = [token.prediction for token in trainer.loop_tokens()]
    return truths, predictions


def benchmark_token_type():
    truths, predictions = predict_token_type_for_benchmark()

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)

    Path("LGBM_results.txt").write_text(f"F1 score {f1}%\n" + f"Accuracy score {accuracy}%")
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    print("start")
    start = time()
    benchmark_token_type()
    print("finished in", int(time() - start), "seconds")
