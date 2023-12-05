import json
import os
import shutil
from collections import Counter
from os.path import join
from pathlib import Path
from time import time

import numpy as np
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfSegment import PdfSegment
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from get_data import get_pdf_name_labels, load_pdf_feature, load_labeled_data, PDF_LABELED_DATA_ROOT_PATH, balance_data
from train_token_type import TOKEN_TYPE_MODEL_PATH
from train_segmentation import SEGMENTATION_MODEL_PATH

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
            'image_id': image_id,
            'bbox': [segment.bounding_box.left, segment.bounding_box.top, segment.bounding_box.width,
                     segment.bounding_box.height],
            'category_id': token_types_to_publaynet_types[segment.segment_type],
            'id': index}


def get_predictions():
    pdf_name_labels = get_pdf_name_labels('dev')
    test_pdf_features = [load_pdf_feature('dev', x) for x in pdf_name_labels if load_pdf_feature('dev', x)]

    print("Predicting token types for", len(test_pdf_features), "pdfs")
    trainer = TokenTypeTrainer(test_pdf_features, ModelConfiguration())
    trainer.set_token_types(TOKEN_TYPE_MODEL_PATH)

    print("Predicting segmentation for", len(test_pdf_features), "pdfs")
    trainer = ParagraphExtractorTrainer(pdfs_features=test_pdf_features, model_configuration=MODEL_CONFIGURATION)
    segments: list[PdfSegment] = trainer.get_pdf_segments(SEGMENTATION_MODEL_PATH)

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


def move_bboxes(prediction_path: str):
    for moving_coordinates in range(1, 3, 1):
        prediction_base = json.loads(Path(prediction_path).read_text())
        for annotation in prediction_base['annotations']:
            annotation['bbox'] = [annotation['bbox'][0] - moving_coordinates, annotation['bbox'][1] - moving_coordinates,
                                  annotation['bbox'][2], annotation['bbox'][3] - moving_coordinates]

        Path(f"data/publaynet/predictions_moving_coordinates_{moving_coordinates}.json").write_text(
            json.dumps(prediction_base))


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


def save_mistakes(truth_path: str, predictions_path: str):
    truth_coco_format = json.loads(Path(truth_path).read_text())
    predictions_format = json.loads(Path(predictions_path).read_text())

    for image in truth_coco_format['images'][:10]:
        copy_pdf_to_mistakes(image)
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_ROOT_PATH, "PubLayNet_LightGBM", image['file_name'])

        annotation_truth_for_image = [x for x in truth_coco_format['annotations'] if x['image_id'] == image['id']]
        annotation_predictions_for_image = [x for x in predictions_format['annotations'] if x['image_id'] == image['id']]

        for annotation in annotation_truth_for_image:
            annotation_rectangle = get_rectangle(annotation)
            prediction_annotations = [x for x in annotation_predictions_for_image if
                                     get_rectangle(x).get_intersection_percentage(annotation_rectangle) > 0]

            metadata = categories[annotation['category_id']].value.lower()

            if not prediction_annotations:
                task_mistakes.add(1, annotation_rectangle, 1, 0, metadata + "pred: void")
                continue

            if prediction_annotations[0]['category_id'] == annotation['category_id']:
                intersection = annotation_rectangle.get_intersection_percentage(get_rectangle(prediction_annotations[0]))
                task_mistakes.add(1, annotation_rectangle, 1, 1, metadata)
                task_mistakes.add(1, get_rectangle(prediction_annotations[0]), 1, 1, f"{intersection:.3}")
                continue

            task_mistakes.add(1, annotation_rectangle, 0, 1, metadata)
            metadata = "pred: " + categories[prediction_annotations[0]['category_id']].value.lower()
            task_mistakes.add(1, get_rectangle(prediction_annotations[0]), 1, 0, metadata)

        task_mistakes.save()


def copy_pdf_to_mistakes(image):
    origin = join("/home/gabo/projects/pdf-token-type-publaynet/data/pdfs/dev", image['file_name'].replace('.jpg', '.pdf'))
    to = join("/home/gabo/projects/pdf-labeled-data/pdfs", image['file_name'], "document.pdf")
    os.makedirs(Path(to).parent, exist_ok=True)
    shutil.copyfile(origin, to)


def get_rectangle(annotation):
    return Rectangle.from_width_height(int(annotation['bbox'][0]),
                                       int(annotation['bbox'][1]),
                                       int(annotation['bbox'][2]),
                                       int(annotation['bbox'][3]))


def learn_coco_format():
    coco_file = json.loads(Path("data/publaynet/val.json").read_text())
    print("")


def create_coco_sub_file():
    image_name_image_id = get_image_name_image_id("train")
    chunk = 0
    pdfs_features: list[PdfFeatures] = load_labeled_data(split="train", from_document_count=chunk * 10000,
                                                         to_document_count=(chunk + 1) * 10000)
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


if __name__ == '__main__':
    print("start")
    start = time()
    check_unbalanced_data()
    # map_score(truth_path="data/publaynet/val.json")
    # save_mistakes(truth_path="data/publaynet/val.json", predictions_path="data/publaynet/predictions_moving_coordinates_1.json")
    print("finished in", int(time() - start), "seconds")
