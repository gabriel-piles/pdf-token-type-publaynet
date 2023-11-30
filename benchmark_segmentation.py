from os.path import join
from pathlib import Path
from time import time

from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from sklearn.metrics import f1_score, accuracy_score

from get_data import get_segmentation_labeled_data
from train_segmentation import loop_pdf_paragraph_tokens, SEGMENTATION_MODEL_PATH


def predict_segmentation_for_benchmark(pdf_paragraph_tokens_list: list[PdfParagraphTokens]):
    pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]
    trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=MODEL_CONFIGURATION)
    truths = []

    for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
        truths.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))

    print("predicting")
    trainer.predict(SEGMENTATION_MODEL_PATH)
    predictions = [token.prediction for token in trainer.loop_tokens()]

    return truths, predictions


def benchmark_segmentation():
    pdf_paragraph_tokens_list = get_segmentation_labeled_data(split="dev")
    truths, predictions = predict_segmentation_for_benchmark(pdf_paragraph_tokens_list)

    f1 = round(f1_score(truths, predictions, average="macro") * 100, 2)
    accuracy = round(accuracy_score(truths, predictions) * 100, 2)
    print(f"F1 score {f1}%")
    print(f"Accuracy score {accuracy}%")
    Path(join("results", "results_segmentation.txt")).write_text(f"F1 score {f1}\nAccuracy score {accuracy}%")


if __name__ == "__main__":
    print("start")
    start = time()
    benchmark_segmentation()
    print("finished in", int(time() - start), "seconds")
