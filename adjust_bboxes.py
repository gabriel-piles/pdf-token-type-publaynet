import json
import os
import shutil
from collections import Counter
from os.path import join
from pathlib import Path
from time import time

from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TaskMistakes import TaskMistakes

from calculate_map import get_rectangle, map_score, categories
from get_data import PDF_LABELED_DATA_ROOT_PATH


def copy_pdf_to_mistakes(split: str, image_name: str):
    pdf_name = image_name.replace('.jpg', '') + '.pdf'
    origin = join("/home/gabo/projects/pdf-token-type-publaynet/data/pdfs", split, pdf_name)
    to = join("/home/gabo/projects/pdf-labeled-data/pdfs", image_name, "document.pdf")
    os.makedirs(Path(to).parent, exist_ok=True)
    shutil.copyfile(origin, to)


def save_mistakes(truth_path: str, prediction_path: str):
    truth_coco_format = json.loads(Path(truth_path).read_text())
    predictions_format = json.loads(Path(prediction_path).read_text())

    for image in truth_coco_format['images'][:25]:
        copy_pdf_to_mistakes('train', image)
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


def move_bboxes():
    moving_coordinates = 1
    prediction_base = json.loads(Path("data/publaynet/predictions_chunk_33.json").read_text())
    for annotation in prediction_base['annotations']:
        rectangle = get_rectangle(annotation)
        if 0 < rectangle.area() < 200:
            annotation['bbox'] = [annotation['bbox'][0],
                                  annotation['bbox'][1] - moving_coordinates,
                                  annotation['bbox'][2],
                                  annotation['bbox'][3]]
        if 1000 < rectangle.area() < 200000:
            annotation['bbox'] = [annotation['bbox'][0],
                                  annotation['bbox'][1] - 2,
                                  annotation['bbox'][2],
                                  annotation['bbox'][3]]

    prediction_moved_path = f"data/publaynet/predictions_moving_top_coordinates_{moving_coordinates}.json"
    Path(prediction_moved_path).write_text(
        json.dumps(prediction_base))

    previous = map_score(truth_path="data/publaynet/train_chunk_33.json", prediction_path="data/publaynet/predictions_chunk_33.json")
    map_score(truth_path="data/publaynet/train_chunk_33.json", prediction_path=prediction_moved_path)
    print("previous", previous)


def get_rectangles(coco_format) -> dict[str, list[Rectangle]]:
    rectangles = dict()

    for annotation in coco_format['annotations']:
        rectangles.setdefault(annotation['image_id'], []).append(get_rectangle(annotation))

    return rectangles


def study_differences(differences: list[tuple[float, int]]):
    buckets = [0, 100, 200, 500, 1000, 3000, 5000, 10000, 50000, 100000]

    for bucket_min, bucket_max in zip(buckets, buckets[1:]):
        print(bucket_max)
        count = Counter([x[1] for x in differences if bucket_min <= x[0] <= bucket_max])
        print(count)


def study():
    truth = json.loads(Path("data/publaynet/train_chunk_33.json").read_text())
    prediction = json.loads(Path("data/publaynet/predictions_chunk_33.json").read_text())

    truth_values: dict[str, list[Rectangle]] = get_rectangles(truth)
    prediction_values: dict[str, list[Rectangle]] = get_rectangles(prediction)

    top_differences = list()
    bottom_differences = list()
    for truth_key, truth_rectangles in truth_values.items():
        for prediction_key, prediction_rectangles in prediction_values.items():
            if truth_key != prediction_key:
                continue

            for truth_rectangle in truth_rectangles:
                for prediction_rectangle in prediction_rectangles:
                    intersection = truth_rectangle.get_intersection_percentage(prediction_rectangle)
                    if intersection < 60:
                        continue

                    top_differences.append((truth_rectangle.area(), truth_rectangle.top - prediction_rectangle.top))
                    bottom_differences.append((truth_rectangle.area(), truth_rectangle.bottom - prediction_rectangle.bottom))

    print("\n\n\n\ntops")
    study_differences(top_differences)
    print("\n\n\n\nbottom")
    study_differences(top_differences)


if __name__ == '__main__':
    start = time()
    print("start")
    save_mistakes(truth_path="data/publaynet/train_chunk_33.json", prediction_path="data/publaynet/predictions_chunk_33.json")
    print("finished in", round(time() - start, 1), "seconds")
