from os import listdir
from os.path import join
from pathlib import Path
from time import time

import numpy as np
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION, config_json
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration

from get_data import get_segmentation_labeled_data

import lightgbm as lgb

MAX_DOCUMENTS = 10000

model_configuration = ModelConfiguration(**config_json)
model_configuration.num_boost_round = 200
model_configuration.num_leaves = 100

segmentation_training_data_path = join("data", "training_data", "segmentation", "train")
SEGMENTATION_MODEL_PATH = join(Path(__file__).parent, "model", "segmentation.model")


def loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list: list[PdfParagraphTokens]):
    for pdf_paragraph_tokens in pdf_paragraph_tokens_list:
        for page in pdf_paragraph_tokens.pdf_features.pages:
            if not page.tokens:
                continue
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                yield pdf_paragraph_tokens, token, next_token
            yield pdf_paragraph_tokens, page.tokens[-1], page.tokens[-1]


def train_segmentation():

    print(f"Getting model input")

    labels = np.array([])
    x_train = None

    for folder_name in list(listdir(segmentation_training_data_path)):
        if x_train is None:
            x_train = np.load(join(segmentation_training_data_path, folder_name, "x.npy"))
        else:
            x_train = np.concatenate((x_train, np.load(join(segmentation_training_data_path, folder_name, "x.npy"))), axis=0)

        labels = np.concatenate((labels, np.load(join(segmentation_training_data_path, folder_name, "y.npy"))), axis=0)

    lgb_train = lgb.Dataset(x_train, labels)
    print(f"Training")
    print(str(x_train.shape))
    gbm = lgb.train(model_configuration.dict(), lgb_train)

    print(f"Saving")
    Path(SEGMENTATION_MODEL_PATH).parent.mkdir(exist_ok=True)
    gbm.save_model(SEGMENTATION_MODEL_PATH, num_iteration=gbm.best_iteration)


def cache_training_data():
    for i in range(5):
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


if __name__ == '__main__':
    print("start")
    start = time()
    cache_training_data()
    print("finished in", int(time() - start), "seconds")