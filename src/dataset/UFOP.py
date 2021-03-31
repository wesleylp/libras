import glob
import os
import re

import cv2

from video import Video, VideoInfo

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

        s = self.video_name.split('_')
        self.id = s[0]
        self.category = s[1]
        self.sequence = s[2]

        self.annotation = labels(labels_path)[self.video_name]

        # TODO: implement segmentation loading
        # if segmentation_path is not None:
        #     self.segmentation = Annotation(self.filepath.replace(".avi", ".json"))

    def get_movement_frames(self):
        return self.annotation['frames']

    def get_labels(self):
        return self.annotation['labels']


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
