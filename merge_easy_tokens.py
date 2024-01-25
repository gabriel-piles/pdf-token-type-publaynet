from paragraph_extraction_trainer.Paragraph import Paragraph
from pdf_features.PdfToken import PdfToken

from get_data import get_segmentation_labeled_data


def merge_close_ones(tokens: list[PdfToken]) -> list[tuple[PdfToken, PdfToken]]:
    pass


def tokens_in_same_paragraph(paragraph: Paragraph, first_token: PdfToken, second_token: PdfToken):
    pass


def check_correct(paragraphs: list[Paragraph], tokens_to_merge: list[tuple[PdfToken, PdfToken]]):
    for first_token, second_token in tokens_to_merge:
        for paragraph in paragraphs:
            if tokens_in_same_paragraph(first_token, second_token):
                break


def get_data():
    segmentation = get_segmentation_labeled_data('dev', from_document_count=0, to_document_count=1)
    print("check it")


if __name__ == '__main__':
    get_data()