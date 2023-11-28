import json
import os
import shutil
from os import listdir
from os.path import join
from pathlib import Path
from time import time

from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType
from sklearn.metrics import f1_score, accuracy_score

from get_data import load_random_labeled_data

PDF_LABELED_DATA_ROOT_PATH = "/home/gabo/projects/pdf-labeled-data"


def convert_labels_to_publaynet(pdf_labels):
    for label in pdf_labels.pages[0].labels:
        if label.label_type not in [2, 3, 4, 5, 6]:
            label.label_type = 6

    return pdf_labels


def predict_for_benchmark():
    file_names = listdir("/home/gabo/projects/pdf-token-type-publaynet/data/VGT_predictions")

    test_pdf_features = load_random_labeled_data(split="dev")

    truths = [token.token_type.get_index() for pdf_features in test_pdf_features for page, token in
              pdf_features.loop_tokens()]

    for pdf_feature in test_pdf_features:
        for token, page in pdf_feature.loop_tokens():
            token.token_type = TokenType.TEXT

        if pdf_feature.file_name not in file_names:
            print("THIS SHOULD NOT HAPPEN Error no labeled data")
            continue

        labeled_data_path = join("/home/gabo/projects/pdf-token-type-publaynet/data/VGT_predictions", pdf_feature.file_name)
        pdf_labels = PdfLabels(**json.loads(Path(labeled_data_path).read_text()))
        pdf_labels = convert_labels_to_publaynet(pdf_labels)
        pdf_feature.set_token_types(pdf_labels)

    predictions = [token.token_type.get_index() for pdf_features in test_pdf_features for page, token in
                   pdf_features.loop_tokens()]

    # save_mistakes(test_pdf_features, truths)
    return truths, predictions


def save_mistakes(test_pdf_features, truths):
    token_index = 0
    for pdf_feature in test_pdf_features[:15]:
        origin = join("/home/gabo/projects/pdf-token-type-publaynet/data/pdfs/dev", pdf_feature.file_name + ".pdf")
        to = join("/home/gabo/projects/pdf-labeled-data/pdfs", pdf_feature.file_name, "document.pdf")
        os.makedirs(Path(to).parent, exist_ok=True)
        shutil.copyfile(origin, to)
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_ROOT_PATH, "VGT", pdf_feature.file_name)

        for page, token in pdf_feature.loop_tokens():
            if token.token_type.get_index() == truths[token_index]:
                task_mistakes.add(1, token.bounding_box, truths[token_index], token.token_type.get_index(),
                                  token.token_type.value)
                token_index += 1
                continue

            metadata = f"{TokenType.from_index(truths[token_index]).value} pred: {token.token_type.value}"
            task_mistakes.add(1, token.bounding_box, truths[token_index], token.token_type.get_index(),
                              metadata)
            token_index += 1

        task_mistakes.save()


def benchmark():
    truths, predictions = predict_for_benchmark()

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)

    Path("VGT_results.txt").write_text(f"F1 score {f1}%\n" + f"Accuracy score {accuracy}%")
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")


if __name__ == "__main__":
    print("start")
    start = time()
    benchmark()
    print("finished in", int(time() - start), "seconds")
