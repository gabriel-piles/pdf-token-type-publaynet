
from time import time

from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_token_type_labels.Label import Label
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from get_data import get_pdf_name_labels, load_pdf_feature
from train_token_type import TOKEN_TYPE_MODEL_PATH
from train_segmentation import SEGMENTATION_MODEL_PATH


def get_predictions():
    pdf_name_labels = get_pdf_name_labels('dev', False, 3)
    test_pdf_features = [load_pdf_feature('dev', x) for x in pdf_name_labels if load_pdf_feature('dev', x)]

    print("Predicting token types for", len(test_pdf_features), "pdfs")
    trainer = TokenTypeTrainer(test_pdf_features, ModelConfiguration())
    trainer.set_token_types(TOKEN_TYPE_MODEL_PATH)

    print("Predicting segmentation for", len(test_pdf_features), "pdfs")
    trainer = ParagraphExtractorTrainer(pdfs_features=test_pdf_features, model_configuration=MODEL_CONFIGURATION)
    segments = trainer.get_pdf_segments(SEGMENTATION_MODEL_PATH)

    print("done")


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
    # annotations = json.loads(Path("data/publaynet/val.json").read_text())
    # for a in annotations['annotations']:
    #     a['score'] = 0.1
    #
    # Path("data/publaynet/test_prediction_val.json").write_text(json.dumps(annotations))
    # print()
    ground_truth = COCO("data/publaynet/val.json")
    predictions = COCO("data/publaynet/test_prediction_val.json")

    average_precision_per_category = {}
    for i in range(1, 6):
        print("Category: ", i)
        coco_eval = COCOeval(ground_truth, predictions, iouType='bbox')    # initialize CocoEval object
        coco_eval.params.catIds = [i]

        # run per image evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        average_precision_per_category[i] = coco_eval.stats[5]

    print("average_precision_per_category")
    print(average_precision_per_category)


if __name__ == '__main__':
    print("start")
    start = time()
    map_test()
    print("finished in", int(time() - start), "seconds")
