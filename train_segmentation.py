from os.path import join
from pathlib import Path

from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION

from get_data import get_segmentation_labeled_data

SEGMENTATION_MODEL = join(Path(__file__).parent, "model", "segmentation.model")
MAX_DOCUMENTS = 30000


def loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list: list[PdfParagraphTokens]):
    for pdf_paragraph_tokens in pdf_paragraph_tokens_list:
        for page in pdf_paragraph_tokens.pdf_features.pages:
            if not page.tokens:
                continue
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                yield pdf_paragraph_tokens, token, next_token
            yield pdf_paragraph_tokens, page.tokens[-1], page.tokens[-1]


def train_segmentation():
    Path(SEGMENTATION_MODEL).parent.mkdir(exist_ok=True)

    pdf_paragraph_tokens_list = get_segmentation_labeled_data(split="train", max_documents=MAX_DOCUMENTS)

    pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]
    trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=MODEL_CONFIGURATION)

    labels = []
    for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
        labels.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))

    trainer.train(str(SEGMENTATION_MODEL), labels)


if __name__ == '__main__':
    train_segmentation()