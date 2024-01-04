import json
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from get_data import get_pdf_name_labels, load_pdf_feature, balance_data

token_types_to_publaynet_types = {
    TokenType.TEXT: 1,
    TokenType.TITLE: 2,
    TokenType.LIST: 3,
    TokenType.TABLE: 4,
    TokenType.FIGURE: 5}

categories = {1: TokenType.TEXT,
              2: TokenType.TITLE,
              3: TokenType.LIST,
              4: TokenType.TABLE,
              5: TokenType.FIGURE}


def get_image_name_image_id(split: str = "val"):
    image_name_image_id = {}

    annotations = json.loads(Path(f"data/publaynet/{split}.json").read_text())

    for image in annotations['images']:
        image_name_image_id[image['file_name'].replace(".jpg", "")] = image['id']

    return image_name_image_id


def get_one_annotation(index, image_id, segment: PdfSegment):
    return {'segmentation': [],
            'area': 1,
            'iscrowd': 0,
            'score': 1,
            'image_id': image_id,
            'bbox': [segment.bounding_box.left, segment.bounding_box.top, segment.bounding_box.width,
                     segment.bounding_box.height],
            'category_id': token_types_to_publaynet_types[segment.segment_type],
            'id': index}


def get_predictions():
    pdf_name_labels = get_pdf_name_labels('dev')
    test_pdf_features = [load_pdf_feature('dev', x) for x in pdf_name_labels if load_pdf_feature('dev', x)]

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
    segments: list[PdfSegment] = trainer.get_pdf_segments("model/segmentation_full_data_4.model")

    segments = [s for s in segments if s.segment_type in token_types_to_publaynet_types.keys()]

    predictions_coco_format = json.loads(Path("data/publaynet/val.json").read_text())

    image_name_image_id = get_image_name_image_id()
    annotations = []
    for i, segment in enumerate(segments):
        annotations.append(get_one_annotation(i, image_name_image_id[segment.pdf_name], segment))

    predictions_coco_format['annotations'] = annotations
    Path("data/publaynet/predictions.json").write_text(json.dumps(predictions_coco_format))


def get_box_from_label(label: Label):
    return [label.left, label.top, label.left + label.width, label.top + label.height]


def get_ground_truths():
    pdf_name_labels = get_pdf_name_labels("dev", False, 3)
    truths = list()

    for pdf_name, labels in pdf_name_labels.items():
        boxes = [get_box_from_label(label) for label in labels]
        labels_int_list = [label.label_type for label in labels]
        truths.append({'boxes': boxes, 'labels': labels_int_list})

    return truths


def map_test():
    coco_file = json.loads(Path("data/publaynet/val.json").read_text())
    for index, a in enumerate(coco_file['annotations']):
        a['score'] = 0.1
        a['id'] = index + 1000000000
        a['segmentation'] = []
        a['area'] = 1

    coco_file['annotations'] = list(reversed(coco_file['annotations']))

    Path("data/publaynet/test_prediction_val.json").write_text(json.dumps(coco_file))
    print()
    ground_truth = COCO("data/publaynet/val.json")
    predictions = COCO("data/publaynet/test_prediction_val.json")

    average_precision_per_category = {}
    for i in range(1, 6):
        print("Category: ", i)
        coco_eval = COCOeval(ground_truth, predictions, iouType='bbox')  # initialize CocoEval object
        coco_eval.params.catIds = [i]

        # run per image evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        average_precision_per_category[i] = coco_eval.stats[5]

    print("average_precision_per_category")
    print(average_precision_per_category)


def map_score(truth_path: str, prediction_path: str):
    ground_truth = COCO(truth_path)
    predictions = COCO(prediction_path)

    average_precision_per_category = {}
    for i in range(1, 6):
        print("Category: ", i)
        coco_eval = COCOeval(ground_truth, predictions, iouType='bbox')  # initialize CocoEval object
        coco_eval.params.catIds = [i]

        # run per image evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        average_precision_per_category[i] = coco_eval.stats[5]

    print()
    overall = round(sum(average_precision_per_category.values()) / len(average_precision_per_category.values()), 3)
    print('\t'.join(["overall"] + [categories[x].value.lower() for x in average_precision_per_category.keys()]))
    print('\t'.join([str(overall)] + [str(round(x, 3)) for x in average_precision_per_category.values()]))

    coco_score = sum([x for x in list(average_precision_per_category.values())[:4]]) / 4
    print("coco_score")
    print(coco_score)
    return coco_score


def get_rectangle(annotation):
    return Rectangle.from_width_height(int(annotation['bbox'][0]),
                                       int(annotation['bbox'][1]),
                                       int(annotation['bbox'][2]),
                                       int(annotation['bbox'][3]))


def learn_coco_format():
    coco_file = json.loads(Path("data/publaynet/val.json").read_text())
    print("")


def create_coco_sub_file():
    chunk = 32

    pdf_name_labels = get_pdf_name_labels('train', from_document_count=10000 * chunk, to_document_count=10000*(chunk + 1))
    pdfs_features = [load_pdf_feature('train', x) for x in pdf_name_labels if load_pdf_feature('train', x)]

    image_name_image_id = get_image_name_image_id("train")
    image_ids = {image_name_image_id[p.file_name] for p in pdfs_features}

    ground_truth = json.loads(Path("data/publaynet/train.json").read_text())

    ground_truth['images'] = [x for x in ground_truth['images'] if x['id'] in image_ids]
    ground_truth['annotations'] = [x for x in ground_truth['annotations'] if x['image_id'] in image_ids]

    Path(f"data/publaynet/train_chunk_{chunk}.json").write_text(json.dumps(ground_truth))


def check_unbalanced_data():
    train = np.load("data/training_data/segmentation/val/chunk_0/x.npy")
    labels = np.load("data/training_data/segmentation/val/chunk_0/y.npy")
    count = Counter()
    count.update(labels)
    print(count)
    print(len(count))

    print("train.size")
    print(train.shape)
    print("labels.size")
    print(labels.shape)

    train, labels = balance_data(train, labels)

    count = Counter()
    count.update(labels)
    print(count)

    print("train.size")
    print(train.shape)
    print("labels.size")
    print(labels.shape)


def annotation_to_predictions_coco():
    predictions_dict = json.loads(Path("vgt_result/inference/coco_instances_results.json").read_text())
    truth_dict = json.loads(Path("data/publaynet/val.json").read_text())

    for i, prediction_ann in enumerate(predictions_dict):
        prediction_ann['id'] = i
        prediction_ann['area'] = 1
    prediction = {"images": truth_dict["images"], "annotations": predictions_dict, "categories": truth_dict["categories"]}
    Path("vgt_result/inference/coco_predictions.json").write_text(json.dumps(prediction))


if __name__ == '__main__':
    print("start")
    start = time()
    print("predictions")
    # create_coco_sub_file()
    # get_predictions()
    # map_score(truth_path="data/publaynet/train_chunk_33.json", prediction_path="data/publaynet/predictions_chunk_33.json")
    map_score(truth_path="data/publaynet/val.json", prediction_path="vgt_result/inference/coco_predictions.json")

    print("finished in", int(time() - start), "seconds")
