import pickle

import cv2
import numpy as np
from src.utils.image import crop_person_img


def load_gei(datapath, dim=None, crop_person=False, flatten=True):

    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    X = [data[idx]['sample'].astype('float64') for idx in range(len(data))]

    if crop_person:
        X = [crop_person_img(x).astype('float64') for x in X]

    if dim is not None:
        X = [cv2.resize(x, dim, interpolation=cv2.INTER_CUBIC).astype('float64') for x in X]

    if flatten:
        X = [x.flatten().astype('float64') for x in X]

    X = np.array(X)

    y = np.array([data[idx]['label'] for idx in range(len(data))]) - 1

    return X, y
