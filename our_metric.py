import pickle
import random
from time import time

from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.PageLabels import PageLabels
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from tqdm import tqdm

from adjust_bboxes import copy_pdf_to_mistakes
from get_data import get_segmentation_labeled_data, get_pdf_name_labels, load_pdf_feature, PDF_LABELED_DATA_ROOT_PATH

TRUTH_SEGMENTS_PICKLE_PATH = "model/truth_segments.pickle"
PREDICTION_SEGMENTS_PICKLE_PATH = "model/prediction_segments.pickle"

SCORE_PER_CATEGORY_PICKLE_PATH = "model/score_per_category.pickle"
COUNT_PER_CATEGORY_PICKLE_PATH = "model/count_per_category.pickle"


def get_predictions():
    pdf_name_labels = get_pdf_name_labels("dev")
    test_pdf_features = [load_pdf_feature("dev", x) for x in pdf_name_labels if load_pdf_feature("dev", x)]

    print("Predicting token types for", len(test_pdf_features), "pdfs")
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
    trainer = TokenTypeTrainer(test_pdf_features, model_configuration)
    trainer.set_token_types("model/token_type_full_data.model")

    print("Predicting segmentation for", len(test_pdf_features), "pdfs")

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

    trainer = ParagraphExtractorTrainer(pdfs_features=test_pdf_features, model_configuration=model_configuration)
    prediction_segments: list[PdfSegment] = trainer.get_pdf_segments("model/segmentation_full_data_4.model")

    with open(PREDICTION_SEGMENTS_PICKLE_PATH, mode="wb") as file:
        pickle.dump(prediction_segments, file)


def get_truth_segments():
    truth_paragraphs_tokens: list[PdfParagraphTokens] = get_segmentation_labeled_data("dev")
    truth_segments: list[PdfSegment] = [
        PdfSegment.from_pdf_tokens(paragraph.tokens, paragraph_token.pdf_features.file_name)
        for paragraph_token in truth_paragraphs_tokens
        for paragraph in paragraph_token.paragraphs
    ]

    with open(TRUTH_SEGMENTS_PICKLE_PATH, mode="wb") as file:
        pickle.dump(truth_segments, file)


def show_mistakes():
    with open(TRUTH_SEGMENTS_PICKLE_PATH, "rb") as f:
        truth_segments: list[PdfSegment] = pickle.load(f)

    with open(PREDICTION_SEGMENTS_PICKLE_PATH, "rb") as f:
        prediction_segments: list[PdfSegment] = pickle.load(f)

    truth_segments_per_pdf = dict()
    prediction_segments_per_pdf = dict()

    for truth_segment in truth_segments:
        truth_segments_per_pdf.setdefault(truth_segment.pdf_name, []).append(truth_segment)

    for prediction_segment in prediction_segments:
        prediction_segments_per_pdf.setdefault(prediction_segment.pdf_name, []).append(prediction_segment)

    keys_sample = random.sample(list(truth_segments_per_pdf.keys()), k=30)
    for pdf_name in tqdm(keys_sample):
        copy_pdf_to_mistakes('dev', pdf_name)
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_ROOT_PATH, "PubLayNet_huridocs_metric", pdf_name)

        for truth_segment in truth_segments_per_pdf[pdf_name]:
            annotated = False
            for prediction_segment in prediction_segments_per_pdf[pdf_name]:
                metadata = truth_segment.segment_type.value.lower() + " pred:" + prediction_segment.segment_type.value.lower()

                intersection_percentage = truth_segment.bounding_box.get_intersection_percentage(prediction_segment.bounding_box)
                if truth_segment.segment_type == prediction_segment.segment_type and intersection_percentage == 100:
                    task_mistakes.add(1, truth_segment.bounding_box, 1, 1, metadata)
                    annotated = True
                    break

                if intersection_percentage == 100:
                    task_mistakes.add(1, truth_segment.bounding_box, 0, 1, metadata)
                    annotated = True
                    break

            if not annotated:
                for prediction_segment in prediction_segments_per_pdf[pdf_name]:
                    metadata = truth_segment.segment_type.value.lower() + " pred:" + prediction_segment.segment_type.value.lower()

                    intersection_percentage = truth_segment.bounding_box.get_intersection_percentage(
                        prediction_segment.bounding_box)
                    if intersection_percentage > 0:
                        annotated = True
                        task_mistakes.add(1, truth_segment.bounding_box, 0, 1, "Truth:" + truth_segment.segment_type.value.lower())
                        task_mistakes.add(1, prediction_segment.bounding_box, 0, 1, metadata)

            if not annotated:
                metadata = "missing:" + truth_segment.segment_type.value.lower()
                task_mistakes.add(1, truth_segment.bounding_box, 1, 0, metadata)

        task_mistakes.save()


def save_scores():
    with open(TRUTH_SEGMENTS_PICKLE_PATH, "rb") as f:
        truth_segments: list[PdfSegment] = pickle.load(f)

    with open(PREDICTION_SEGMENTS_PICKLE_PATH, "rb") as f:
        prediction_segments: list[PdfSegment] = pickle.load(f)

    score_per_category = {x: 0 for x in TokenType}
    count_per_category = {x: 0 for x in TokenType}

    error_segmentation = {x: 0 for x in TokenType}
    error_token_type = {x: 0 for x in TokenType}

    truth_segments_per_pdf = dict()
    prediction_segments_per_pdf = dict()

    for truth_segment in truth_segments:
        truth_segments_per_pdf.setdefault(truth_segment.pdf_name, []).append(truth_segment)

    for prediction_segment in prediction_segments:
        prediction_segments_per_pdf.setdefault(prediction_segment.pdf_name, []).append(prediction_segment)

    for pdf_name, truth_segments in tqdm(truth_segments_per_pdf.items()):
        for truth_segment in truth_segments:
            for prediction_segment in prediction_segments_per_pdf[truth_segment.pdf_name]:
                intersection_percentage = truth_segment.bounding_box.get_intersection_percentage(prediction_segment.bounding_box)

                if truth_segment.segment_type == prediction_segment.segment_type and intersection_percentage == 100:
                    score_per_category[truth_segment.segment_type] += 1
                    break

                if intersection_percentage == 100 and truth_segment.segment_type != prediction_segment.segment_type:
                    error_token_type[truth_segment.segment_type] += 1
                    break

                error_segmentation[truth_segment.segment_type] += 1

            count_per_category[truth_segment.segment_type] += 1

    print("error_segmentation")
    print({x.value: v for x,v in error_segmentation.items() if v != 0})
    print("error_token_type")
    print({x.value: v for x,v in error_token_type.items() if v != 0})
    # with open(SCORE_PER_CATEGORY_PICKLE_PATH, mode="wb") as file:
    #     pickle.dump(score_per_category, file)
    #
    # with open(COUNT_PER_CATEGORY_PICKLE_PATH, mode="wb") as file:
    #     pickle.dump(count_per_category, file)


def print_scores():
    with open(SCORE_PER_CATEGORY_PICKLE_PATH, "rb") as f:
        score_per_category: dict[TokenType, int] = pickle.load(f)

    with open(COUNT_PER_CATEGORY_PICKLE_PATH, "rb") as f:
        count_per_category: dict[TokenType, int] = pickle.load(f)

    accuracies = list()
    for token_type in TokenType:
        if count_per_category[token_type]:
            accuracy = round(100 * score_per_category[token_type] / count_per_category[token_type], 2)
            accuracies.append(accuracy)
            print(token_type.value, f"{accuracy}%")

    print('Average', f"{round(sum(accuracies)/len(accuracies), 2)}%")


def one_document_get_segmentation_labeled_data(pdf_name: str) -> list[PdfParagraphTokens]:
    all_pdf_name_labels = get_pdf_name_labels('dev', True, 0, 99999999)
    labels = [labels for key_pdf_name, labels in all_pdf_name_labels.items() if key_pdf_name == pdf_name][0]

    pdfs_paragraphs_tokens: list[PdfParagraphTokens] = list()

    pdf_feature = load_pdf_feature('dev', pdf_name)

    pages = [PageLabels(number=1, labels=labels)]
    pdf_paragraphs_tokens = PdfParagraphTokens.set_paragraphs(pdf_feature, PdfLabels(pages=pages))
    pdfs_paragraphs_tokens.append(pdf_paragraphs_tokens)
    # Label(top=99, left=307, width=242, height=187, label_type=3, metadata='')
    # Label(top=281, left=312, width=182, height=10, label_type=6, metadata='')
    #<text top="281" left="309" width="4" height="6" font="6"><i>∗</i></text>
    return pdfs_paragraphs_tokens


if __name__ == "__main__":
    start = time()
    print("start")
    # get_predictions()
    # print_scores()
    # show_mistakes()
    # pdf_features = load_pdf_feature("dev", "PMC3170864_00003")
    segmentations = one_document_get_segmentation_labeled_data("PMC3170864_00003")
    # test_segmentation = [x for x in segmentations if x.pdf_features.file_name == "PMC3170864_00003"]
    print("finished in", round(time() - start, 1), "seconds")
