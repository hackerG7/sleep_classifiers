from io import StringIO
from pathlib import Path

import numpy as np
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_classifiers():
    return [AttributedClassifier(name='Random Forest',
                                 classifier=RandomForestClassifier(n_estimators=100, max_features=1.0,
                                                                   max_depth=10,
                                                                   min_samples_split=10, min_samples_leaf=32,
                                                                   bootstrap=True)),
            AttributedClassifier(name='Logistic Regression',
                                 classifier=LogisticRegression(penalty='l1', solver='liblinear', verbose=0,
                                                               multi_class='auto')),
            AttributedClassifier(name='k-Nearest Neighbors',
                                 classifier=KNeighborsClassifier(weights='distance')),
            AttributedClassifier(name='Neural Net',
                                 classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                                                          max_iter=2000, alpha=0.01, solver='adam', verbose=False,
                                                          n_iter_no_change=20))]


def get_base_feature_sets():
    return [#[FeatureType.count],
            #[FeatureType.heart_rate],
            [FeatureType.count, FeatureType.heart_rate]
    ]
    # 12-23-19 note: I'm making the default base feature use cosine, not circadian model
    # so that it doesn't require MATLAB to run


def convert_pdf_to_txt(pdf_path_string, all_texts):
    resource_manager = PDFResourceManager()
    returned_string = StringIO()
    codec = 'utf-8'
    layout_parameters = LAParams(all_texts=all_texts)
    device = TextConverter(resource_manager, returned_string, codec=codec, laparams=layout_parameters)
    fp = open(pdf_path_string, 'rb')
    interpreter = PDFPageInterpreter(resource_manager, device)
    password = ""
    max_pages = 0
    caching = True
    page_numbers = set()

    for page in PDFPage.get_pages(fp, page_numbers, maxpages=max_pages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = returned_string.getvalue()

    fp.close()
    device.close()
    returned_string.close()
    return text


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value


def convolve_with_dog(y, box_pts):
    print("box_pts: ",box_pts)
    print("y: ", y)
    y = y - np.mean(y)
    print("new y: ", y)
    box = np.ones(box_pts) / box_pts
    print("box: ", box)

    mu1 = int(box_pts / 2.0)
    print("mu1: ", mu1)
    sigma1 = 120
    print("sigma1: ", sigma1)

    mu2 = int(box_pts / 2.0)
    print("mu2: ", mu2)
    sigma2 = 600
    print("sigma2: ", sigma2)

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))
    print("box_pts: ",box_pts)
    print("int(box_pts / 2): ",int(box_pts / 2))
    print("new box: ", box)
    print("y[0:int(box_pts / 2)].shape: ", y[0:int(box_pts / 2)].shape)
    print("y[0:int(box_pts / 2)]: ", y[0:int(box_pts / 2)])
    print("np.flip(y[0:int(box_pts / 2)])", np.flip(y[0:int(box_pts / 2)]))
    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions

    print("new new y: ", y)
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    print("new new new y.shape: ", y.shape)
    print("new new new y: ", y)
    print("box: ", box)
    y_smooth = np.convolve(y, box, mode='valid')
    print("y_smooth.shape: ", y_smooth.shape)
    print("y_smooth: ", y_smooth)

    return y_smooth


def remove_repeats(array):
    array_no_repeats = np.unique(array, axis=0)
    array_no_repeats = array_no_repeats[np.argsort(array_no_repeats[:, 0])]
    return array_no_repeats


def remove_nans(array):
    array = array[~np.isnan(array).any(axis=1)]
    array = array[~np.isinf(array).any(axis=1)]
    return array
#print("cwd", convolve_with_dog([80.5, 70.3, 60.5, 65.5], 10 * 30 - 15))