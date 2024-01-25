import json
import pickle
from pathlib import Path

from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.TaskMistakes import TaskMistakes
from pdf_token_type_labels.TokenType import TokenType
from tqdm import tqdm

from get_data import publaynet_types_to_token_types, load_pdf_feature, PDF_LABELED_DATA_ROOT_PATH
from our_metric import PREDICTION_SEGMENTS_PICKLE_PATH
from pdfs_in_labeled_data import pdfs_in_labeled_data

MISTAKES_NAME = "VGT_PubLayNet"


def get_segments_from_labels(pdf_name_labels, test_pdf_features):
    label_segments: list[PdfSegment] = list()
    for pdf_features in test_pdf_features:
        file_name = pdf_features.file_name
        for page in pdf_features.pages:
            labels = pdf_name_labels[f"{file_name}"]
            for label in labels:
                label_segments.append(
                    PdfSegment(page_number=page.page_number,
                               bounding_box=get_rectangle_from_label(label),
                               text_content=label.metadata,
                               segment_type=TokenType.from_index(label.label_type),
                               pdf_name=pdf_features.file_name))

    print("sorting")
    label_segments = sorted(label_segments, key=lambda x: float(x.text_content), reverse=True)
    print("sorted")
    return label_segments


def get_rectangle_from_label(label):
    label_rectangle = Rectangle.from_width_height(label.left, label.top, label.width, label.height)
    return label_rectangle


def get_most_probable(pdfs_features: list[PdfFeatures], labels_pdfs_segments: list[PdfSegment]):
    label_list_token: dict[PdfSegment, list[PdfToken]] = dict()
    for pdf_features in tqdm(pdfs_features):
        label_pdf_segment = [x for x in labels_pdfs_segments if x.pdf_name == pdf_features.file_name]
        for page, token in pdf_features.loop_tokens():
            best_score: float = 0
            best_label: PdfSegment | None = None
            for label in label_pdf_segment:
                if float(label.text_content) > best_score and label.bounding_box.get_intersection_percentage(token.bounding_box):
                    best_score = float(label.text_content)
                    best_label = label
                    if best_score >= 0.99:
                        break

            if best_label:
                label_list_token.setdefault(best_label, list()).append(token)
            else:
                token.content = "0"
                token.token_type = TokenType.HEADER
                label_list_token[PdfSegment.from_pdf_tokens([token], pdf_features.file_name)] = [token]

    return label_list_token


def get_pdf_features(pdf_name_labels):
    pdfs_features: list[PdfFeatures] = list()
    for pdf_name, labels in pdf_name_labels.items():
        pdf_feature = load_pdf_feature("dev", pdf_name)

        if not pdf_feature:
            continue
        pdfs_features.append(pdf_feature)

    return pdfs_features


def get_labels():
    json_path = "data/publaynet/val.json"
    source_labels = json.loads(Path(json_path).read_text())
    images_names = {value['id']: value['file_name'] for value in source_labels['images']}
    prediction_annotations = json.loads(Path("vgt_result/inference/coco_predictions.json").read_text())
    pdf_name_labels = dict()
    for annotation in prediction_annotations["annotations"]:
        pdf_name = images_names[annotation['image_id']][:-4]

        category_id = publaynet_types_to_token_types[annotation['category_id']]
        label = Label(left=int(annotation['bbox'][0]),
                      top=int(annotation['bbox'][1]),
                      width=int(annotation['bbox'][2]),
                      height=int(annotation['bbox'][3]),
                      label_type=TokenType.from_text(category_id).get_index(),
                      metadata=str(annotation['score']))

        pdf_name_labels.setdefault(pdf_name, list()).append(label)
    return pdf_name_labels


def get_vgt_predictions():
    pdf_name_labels = get_labels()
    print("total", len(pdf_name_labels))

    pdfs_features = get_pdf_features(pdf_name_labels)
    label_segments = get_segments_from_labels(pdf_name_labels, pdfs_features)

    label_list_token: dict[PdfSegment, list[PdfToken]] = get_most_probable(pdfs_features, label_segments)

    most_probable_pdf_segments: list[PdfSegment] = list()
    for label, tokens_list in label_list_token.items():
        prediction_pdf_segment = PdfSegment.from_pdf_tokens(tokens_list, label.pdf_name)
        prediction_pdf_segment.text_content = label.text_content
        prediction_pdf_segment.segment_type = label.segment_type
        most_probable_pdf_segments.append(prediction_pdf_segment)

    with open(PREDICTION_SEGMENTS_PICKLE_PATH, mode="wb") as file:
        pickle.dump(most_probable_pdf_segments, file)


def show_predictions():
    pdf_name_labels = get_labels()
    pdfs_features = get_pdf_features(pdf_name_labels)
    pdfs_features = [x for x in pdfs_features if x.file_name in pdfs_in_labeled_data][:20]
    label_segments = get_segments_from_labels(pdf_name_labels, pdfs_features)

    label_list_token: dict[PdfSegment, list[PdfToken]] = get_most_probable(pdfs_features, label_segments)

    most_probable_pdf_segments: list[PdfSegment] = list()
    for label, tokens_list in label_list_token.items():
        prediction_pdf_segment = PdfSegment.from_pdf_tokens(tokens_list, label.pdf_name)
        prediction_pdf_segment.text_content = label.text_content
        prediction_pdf_segment.segment_type = label.segment_type
        most_probable_pdf_segments.append(prediction_pdf_segment)

    for pdf_features in pdfs_features:
        task_mistakes = TaskMistakes(PDF_LABELED_DATA_ROOT_PATH, MISTAKES_NAME, pdf_features.file_name)

        for segment in [x for x in most_probable_pdf_segments if x.pdf_name == pdf_features.file_name]:
            metadata = f"{segment.segment_type.value.lower()} {segment.text_content}"
            task_mistakes.add(segment.page_number, segment.bounding_box, 1, 1, metadata)

        task_mistakes.save()


if __name__ == '__main__':
    get_vgt_predictions()
    # show_predictions()
