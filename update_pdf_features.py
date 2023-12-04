import pickle
from os import listdir
from os.path import join
from time import time

from pdf_features.PdfFeatures import PdfFeatures


def update():
    path = "data/pdf_features/train"
    for index, file_name in enumerate(listdir(path)):

        print("updating", join(path, file_name))
        with open(join(path, file_name), "rb") as f:
            pdf_features: PdfFeatures = pickle.load(f)

        pdf_features.pages[0].pdf_name = pdf_features.file_name
        print(pdf_features)

        with open(join(path, file_name), "wb") as file:
            pickle.dump(pdf_features, file)


if __name__ == '__main__':
    start = time()
    print("start")
    update()
    print("finished in", round(time() - start, 1), "seconds")


