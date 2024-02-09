import unicodedata

from tqdm import tqdm

from get_data import load_labeled_data


def run():
    categories = set()
    for i in range(15):
        train_pdf_features = load_labeled_data(split="train", from_document_count=5000*i, to_document_count=5000*(i+1))
        for pdf_features in tqdm(train_pdf_features):
            for page, token in pdf_features.loop_tokens():
                for letter in token.content:
                    categories.add(unicodedata.category(letter))

    print(categories)


if __name__ == '__main__':
    run()