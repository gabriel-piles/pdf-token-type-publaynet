import json
import os
import pickle
import random
import shutil
from os import listdir
from os.path import join, exists
from pathlib import Path
from time import time

from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.PageLabels import PageLabels
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType

publaynet_types_to_token_types = {
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"}


PDF_LABELED_DATA_ROOT_PATH = "/home/gabo/projects/pdf-labeled-data"


def check_data():
    json_path = Path("/home/gabo/projects/pdf-token-type-publaynet/data/publaynet/val.json")
    validation_json = json.loads(json_path.read_text())

    # print(validation_json["images"][0])
    # dict_keys(['images', 'annotations', 'categories'])
    print(validation_json["annotations"][1])
    # {'iscrowd': 0, 'image_id': 341427, 'bbox': [308.61, 490.87, 240.19, 252.16], 'category_id': 1, 'id': 3322349}


def get_token_types():
    for i in range(len(TokenType)):
        print(list(TokenType)[i].value)


def cache_pdf_features_from_path_labels(split: str, pdf_name_labels: dict[str, list[Label]]):
    cache_pdf_features_path = join(".", "data", "pdf_features", "train" if split == "train" else "dev")
    pdf_folder_path = join(".", "data", "pdfs", "train" if split == "train" else "dev")

    for pdf_name, labels in pdf_name_labels.items():
        pickle_path = join(cache_pdf_features_path, pdf_name + ".pickle")
        if exists(pickle_path):
            continue

        print("caching ", pdf_name)
        pdf_features = PdfFeatures.from_pdf_path(join(pdf_folder_path, pdf_name + ".pdf"))

        if not pdf_features:
            continue

        pdf_features.file_name = pdf_name
        pdf_labels = PdfLabels(pages=[PageLabels(number=1, labels=labels)])
        pdf_features.set_token_types(pdf_labels)
        with open(pickle_path, "wb") as file:
            pickle.dump(pdf_features, file)


def get_pdf_name_labels(split: str, max_documents: int = 99999999) -> dict[str, list[Label]]:
    json_path = "data/publaynet/" + ("train" if split == "train" else "val") + ".json"
    source_labels = json.loads(Path(json_path).read_text())

    images_names = {value['id']: value['file_name'] for value in source_labels['images']}

    pdf_name_labels = dict()
    for annotation in source_labels["annotations"]:
        pdf_name = images_names[annotation['image_id']][:-4]

        if len(pdf_name_labels) >= max_documents and pdf_name not in pdf_name_labels:
            continue

        category_id = publaynet_types_to_token_types[annotation['category_id']]
        label = Label(left=int(annotation['bbox'][0]) - 1,
                      top=int(annotation['bbox'][1]) - 1,
                      width=int(annotation['bbox'][2]) + 2,
                      height=int(annotation['bbox'][3]) + 2,
                      label_type=TokenType.from_text(category_id).get_index())

        pdf_name_labels.setdefault(pdf_name, list()).append(label)

    return pdf_name_labels


def cache_pdf_features(split: str, max_documents: int = 99999999):
    pdf_name_labels = get_pdf_name_labels(split, max_documents)
    cache_pdf_features_from_path_labels(split, pdf_name_labels)


def load_random_labeled_data(split: str, max_documents: int = 99999999):
    cache_pdf_features_path = join(".", "data", "pdf_features", "train" if split == "train" else "dev")
    pdfs_features: list[PdfFeatures] = list()

    all_files = listdir(cache_pdf_features_path)

    files = all_files if len(all_files) <= max_documents else random.sample(all_files, max_documents)

    for file in files:
        with open(join(cache_pdf_features_path, file), "rb") as f:
            pdfs_features.append(pickle.load(f))

    return pdfs_features


def load_pdf_feature(split: str, pdf_name: str):
    cache_pdf_features_path = join(".", "data", "pdf_features", "train" if split == "train" else "dev")

    if not exists(join(cache_pdf_features_path, pdf_name + '.pickle')):
        return None

    with open(join(cache_pdf_features_path, pdf_name + '.pickle'), "rb") as f:
        return pickle.load(f)


def get_segmentation_labeled_data(split: str, max_documents: int = 99999999) -> list[PdfParagraphTokens]:
    pdf_name_labels = get_pdf_name_labels(split, max_documents)
    pdfs_paragraphs_tokens: list[PdfParagraphTokens] = list()
    for pdf_name, labels in pdf_name_labels.items():
        pdf_feature = load_pdf_feature(split, pdf_name)

        if not pdf_feature:
            continue

        pages = [PageLabels(number=1, labels=labels)]
        pdf_paragraphs_tokens = PdfParagraphTokens.set_paragraphs(pdf_feature, PdfLabels(pages=pages))
        pdfs_paragraphs_tokens.append(pdf_paragraphs_tokens)

    return pdfs_paragraphs_tokens


def show_segmentation():
    pdf_path_labels = get_pdf_name_labels('dev', 20)
    for file_name, labels in pdf_path_labels.items():
        origin = join("/home/gabo/projects/pdf-token-type-publaynet/data/pdfs/dev", file_name + ".pdf")
        to = join("/home/gabo/projects/pdf-labeled-data/pdfs", file_name, "document.pdf")
        os.makedirs(Path(to).parent, exist_ok=True)
        shutil.copyfile(origin, to)
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_ROOT_PATH, "PubLayNet_segmentation", file_name)

        for label in labels:
            metadata = TokenType.from_index(label.label_type).value
            bounding_box = Rectangle.from_width_height(label.left,label.top, label.width, label.height)
            task_mistakes.add(1, bounding_box, 1, 1, metadata)

        task_mistakes.save()


if __name__ == '__main__':
    start = time()
    print("start")
    print("finished in", round(time() - start, 1), "seconds")
    print()

