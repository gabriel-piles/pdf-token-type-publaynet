import json
from pathlib import Path

from pdf_features.PdfFeatures import PdfFeatures
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.PageLabels import PageLabels
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TokenType import TokenType

publaynet_types_to_token_types = {
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"}


def check_data():
    json_path = Path("/home/gabo/projects/pdf-token-type-publaynet/data/publaynet-labels/val.json")
    validation_json = json.loads(json_path.read_text())
    # print(validation_json.keys())
    # dict_keys(['images', 'annotations', 'categories'])
    # print(validation_json["images"][0])
    print(validation_json["annotations"][1])
    # {'iscrowd': 0, 'image_id': 341427, 'bbox': [308.61, 490.87, 240.19, 252.16], 'category_id': 1, 'id': 3322349}


def get_token_types():
    for i in range(len(TokenType)):
        print(list(TokenType)[i].value)


def load_labeled_data(split: str) -> list[PdfFeatures]:
    if split == "train":
        json_path = "data/publaynet-labels/train.json"
    else:
        json_path = "data/publaynet-labels/val.json"

    pdf_path_labels = get_path_labels(json_path)

    return get_pdf_features_from_path_lables(pdf_path_labels, pdfs_features)


def get_pdf_features_from_path_lables(pdf_path_labels, pdfs_features):
    for pdf_path, labels in pdf_path_labels.items():
        pdf_features = PdfFeatures.from_pdf_path(pdf_path)
        pdf_labels = PdfLabels(pages=[PageLabels(number=1, labels=labels)])
        pdf_features.set_token_types(pdf_labels)
        pdfs_features.append(pdf_features)


def get_path_labels(json_path: str):
    source_labels = json.loads(Path(json_path).read_text())

    images_paths = {value['id']: value['file_name'] for value in source_labels['images']}

    pdf_path_labels = dict()
    for annotation in source_labels["annotations"]:
        pdf_path = images_paths[annotation['image_id']]
        label = Label(left=int(annotation['bbox'][0]),
                      top=int(annotation['bbox'][1]),
                      width=int(annotation['bbox'][2]),
                      height=int(annotation['bbox'][3]),
                      label_type=TokenType.from_text(publaynet_types_to_token_types[annotation['category_id']]))

        pdf_path_labels.setdefault(pdf_path, list()).append(label)

    return pdf_path_labels


if __name__ == "__main__":
    # print(TokenType.from_text("text"))
    # check_data()
    load_labeled_data('val')
