import glob
import itertools
import os
import pickle
import re
import sys

import cv2
import numpy as np
import sklearn

from .annotation import Annotation
from .video import Video, VideoInfo

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../detectron2/projects/DensePose/'))

from src.utils.image import crop_person_img

# it seems there is a mistake in the classes names from class 20 and so
# on
classes = {
    1: "Year 1",
    2: "Year 2",
    3: "Year 3",
    4: "Day 1",
    5: "Day 2",
    6: "Day 3",
    7: "Week 1",
    8: "Week",
    9: "Yesterday",
    10: "Day before Yesterday",
    11: "Safe",
    12: "Physiotherapy",
    13: "Idea",
    14: "Stamp",
    15: "Record",
    16: "Effort",
    17: "Defend",
    18: "Physical education",
    19: "Bodybuilding",
    20: "Battle",
    21: "Close",
    22: "Screw up",
    23: "Bicycle",
    24: "Slip",
    25: "Always",
    26: "Build",
    27: "Calumny",
    28: "Work",
    29: "Television",
    30: "Love",
    31: "Learn",
    32: "Analyze",
    33: "Talk",
    34: "Cock",
    35: "Hen",
    36: "Interact",
    37: "Exchange",
    38: "Strong wind",
    39: "Weak wind",
    40: "Strong rain",
    41: "Weak rain",
    42: "Run fast",
    43: "Run slow",
    44: "Takes great care",
    45: "Takes a little care",
    46: "Thin",
    47: "Fat",
    48: "Strong",
    49: "Weak",
    50: "Arrive",
    51: "Win",
    52: "Loss",
    53: "Open",
    54: "Nothing",
    55: "Nobody",
    56: "No",
}

category = {
    "C1": [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8),
        (9, 10),
    ],
    "C2": [
        (11, 12),
        (13, 4),
        (14, 15),
        (16, 17),
        (18, 19),
        (20, 21),
        (22, 23),
        (24, 25),
        (26, 27),
        (28, 29),
    ],
    "C3": [
        (30, 31),
        (32, 33),
        (34, 35),
        (36, 37),
    ],
    "C4": [
        (38, 39),
        (40, 41),
        (42, 43),
        (44, 45),
        (46, 47),
        (48, 49),
        (50, 51, 52, 53, 21),
        (54, 55, 56),
    ],
}

# experimental set on the LIBRAS-UFOP datase
exp_set = {
    'set_1': {
        'train': (1, 2, 3),
        'val': (4, ),
        'test': (5, )
    },
    'set_2': {
        'train': (2, 3, 4),
        'val': (5, ),
        'test': (1, )
    },
    'set_3': {
        'train': (1, 4, 5),
        'val': (2, ),
        'test': (3, )
    },
    'set_4': {
        'train': (1, 2, 5),
        'val': (3, ),
        'test': (4, )
    },
    'set_5': {
        'train': (3, 4, 5),
        'val': (1, ),
        'test': (2, )
    },
}


class UFOPVideo(Video):
    def __init__(self, filepath, labels_path, segmentation_path=None):
        self.filepath = filepath
        p = re.compile(r'p[0-9]*_c[0-9]*_s[0-9]*')
        self.video_name = p.findall(self.filepath)[0]
        self.capture = cv2.VideoCapture(self.filepath)
        self.video_info = VideoInfo(self.filepath)

        self.annotation = labels(labels_path)[self.video_name]

        s = self.video_name.split('_')
        self.id = int(s[0][1:])
        self.category = s[1]
        self.sequence = s[2]
        self.word = self.get_word()
        self.sign_id = self.get_sign_id()

        if segmentation_path is not None:
            segm_path = self.filepath.replace('LIBRAS-UFOP', 'LIBRAS-UFOP/segm')
            self.segmentation = Annotation(segm_path.replace(".avi", ".json"))

    def get_movement_frames(self):
        return self.annotation['frames']

    def get_labels(self):
        return self.annotation['labels']

    def get_sign_id(self):
        sign_id = np.unique(self.get_labels())
        assert len(sign_id) == 1, "Different labels in the same video"

        # this is a `gambiarra` to fix the class 20 missing in the dataset
        if sign_id > 20:
            sign_id -= 1

        return int(sign_id)

    def get_word(self):
        return classes[self.get_sign_id()]


class UFOPDataset(object):
    def __init__(self, rootdir="../data/LIBRAS-UFOP", labels_path="../data/LIBRAS-UFOP/labels.txt"):
        self._rootdir = rootdir
        self.filepaths = self.get_files()
        self.__labels_path = labels_path
        self._index = 0

    def __next__(self):
        if self._index < len(self.filepaths):
            video = UFOPVideo(self.filepaths[self._index], self.__labels_path)
            self._index += 1
            return video
        else:
            raise StopIteration

    def __len__(self):
        return len(self.filepaths)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return UFOPVideo(self.filepaths[idx], self.__labels_path)

    def get_files(self):
        list_of_videos = (glob.glob(os.path.join(self._rootdir, '**/*Color.avi'), recursive=True))
        return list_of_videos

    def _load_gei(self, subset='set_1', mode='train', categ="all"):
        """Load the GEI features

        Args:
            subset (str, optional): select wich one of the 3 subsets will be loaded. Possible values: `set_n`, where n = {1,2,3}. Defaults to 'set_1'.
            mode (str, optional): Select `train`, `val` or `test` set. Defaults to 'train'.
            categ (str, optional): Select which category (see paper) `c_n`, where n={1,2,3,4} to be loaded or `all` to load all categories.  Defaults to "all".
        """
        def load_geis_in_path(datapath):
            geis_paths = glob.glob(os.path.join(datapath, '*.pkl'), recursive=True)
            data = []
            for gei_path in geis_paths:
                with open(gei_path, 'rb') as f:
                    data.append(pickle.load(f))
            return data

        sinalizers_config = exp_set[subset][mode]

        if categ.lower() != "all":
            words_id_config = list(itertools.chain(*category[categ.upper()]))
            list_of_words = [classes[w] for w in words_id_config]

        else:
            list_of_words = list(classes.values())

        # include similar word in c2 category
        if categ.lower() == 'c2':
            list_of_words.append(classes[4])

        # include similar word in c4 category
        if categ.lower() == 'c4':
            list_of_words.append(classes[21])

        data = []
        label = []
        for vid_path in self.filepaths:
            video = UFOPVideo(vid_path, self.__labels_path)

            if video.id not in sinalizers_config or video.word not in list_of_words:
                continue

            else:
                geis_path = os.path.dirname(video.filepath).replace("LIBRAS-UFOP",
                                                                    "LIBRAS-UFOP/gei")
                geis = load_geis_in_path(geis_path)
                lbl = len(geis) * [video.sign_id]  # could be video.word

                data.extend(geis)
                label.extend(lbl)

        return data, label

    def load_features(self,
                      subset='set_1',
                      mode='train',
                      categ="all",
                      dim=None,
                      crop_person=False,
                      shuffle=True,
                      flatten=True):

        X, y = self._load_gei(subset=subset, mode=mode, categ=categ)

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y, random_state=0)

        if crop_person:
            X = [crop_person_img(x).astype('float64') for x in X]

        if dim is not None:
            X = [cv2.resize(x, dim, interpolation=cv2.INTER_CUBIC).astype('float64') for x in X]

        if flatten:
            X = [x.flatten().astype('float64') for x in X]

        X = np.array(X)
        y = np.array(y)

        return X, y


def gen_cv(y_train, y_valid):
    "Fix the validation set to be use in cross-validation"

    train_indexes = -np.ones(y_train.shape[0])
    valid_indexes = np.zeros(y_valid.shape[0])

    all_indexes = np.concatenate((train_indexes, valid_indexes))

    return sklearn.model_selection.PredefinedSplit(test_fold=all_indexes)


def labels(labels_path):

    annot = {}

    with open(labels_path) as fp:
        lines = fp.readlines()

        for line in lines:
            lin = line.strip('\n').split(' ')

            fold = lin[0]
            frames_labels = lin[1:]

            annot[fold] = {
                'labels': [],
                'frames': [],
            }

            for fl in frames_labels:
                frames = fl.split(':')[0]
                frames = frames.split('-')
                annot[fold]['frames'].append((int(frames[0]), int(frames[1])))

                label = fl.split(':')[-1]
                annot[fold]['labels'].append(int(label))

    return annot


if __name__ == "__main__":
    annot = labels("/home/wesley.passos/repos/libras/data/LIBRAS-UFOP/labels.txt")

    data = UFOPDataset("/home/wesley.passos/repos/libras/data/LIBRAS-UFOP",
                       "/home/wesley.passos/repos/libras/data/LIBRAS-UFOP/labels.txt")

    a, b = data.load_features(categ='c2')
    print(a)
