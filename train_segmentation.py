
from os.path import join
from pathlib import Path
from time import time
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

from calculate_map import map_score
from get_data import get_segmentation_labeled_data


from hyperparams import set_segmentation_predictions, PREDICTIONS_PATH
from train import train

MAX_DOCUMENTS = 10000

SEGMENTATION_MODEL_PATH = join(Path(__file__).parent, "model", "segmentation.model")

configuration_dict = dict()
configuration_dict["context_size"] = 1
configuration_dict["num_boost_round"] = 500
configuration_dict["num_leaves"] = 500
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


def train_segmentation():
    train(model_configuration=model_configuration,
          training_data_path="data/training_data/segmentation/train",
          model_path=SEGMENTATION_MODEL_PATH,
          chunks_count=33)


def cache_segmentation_training_data():
    for i in range(8):
        start_cache = time()
        print(f"Caching chunk {i}")
        pdf_paragraph_tokens_list = get_segmentation_labeled_data(split="train", from_document_count=MAX_DOCUMENTS * i,
                                                                  to_document_count=MAX_DOCUMENTS * (i + 1))

        if not pdf_paragraph_tokens_list:
            continue

        pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]
        trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=MODEL_CONFIGURATION)
        labels = []
        for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
            labels.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))

        trainer.save_training_data(join("data", "training_data", "segmentation", "train", f"chunk_{i}"), labels)
        print("finished in", round(time() - start_cache, 1), "seconds\n")


def cache_validation_data():
    pdf_paragraph_tokens_list = get_segmentation_labeled_data(split="dev", from_document_count=0, to_document_count=999999999)
    
    pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]
    trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=MODEL_CONFIGURATION)

    labels = []
    for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
        labels.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))
        
    trainer.save_training_data(join("data", "training_data", "segmentation", "val", "chunk_0"), labels)


def evaluate_results():
    test_chunk = 33
    set_segmentation_predictions(model_configuration=model_configuration,
                                 segmentation_model_path=SEGMENTATION_MODEL_PATH,
                                 chunk=test_chunk)

    coco_score = map_score(truth_path=f"data/publaynet/train_chunk_{test_chunk}.json", prediction_path=PREDICTIONS_PATH)

    print("coco_score")
    print(coco_score)


if __name__ == '__main__':
    print("start")
    start = time()
    train_segmentation()
    print("finished in", int(time() - start), "seconds")