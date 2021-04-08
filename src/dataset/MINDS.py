import glob
import os
import pickle
import sys

import cv2
import numpy as np
import sklearn

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../detectron2/projects/DensePose/'))

from .utils.image import crop_person_img
from .video import Video, VideoInfo

classes = {
    1: "Acontecer",
    2: "Aluno",
    3: "Amarelo",
    4: "America",
    5: "Aproveitar",
    6: "Bala",
    7: "Banco",
    8: "Banheiro",
    9: "Barulho",
    10: "Cinco",
    11: "Conhecer",
    12: "Espelho",
    13: "Esquina",
    14: "Filho",
    15: "Maca",
    16: "Medo",
    17: "Ruim",
    18: "Sapo",
    19: "Vacina",
    20: "Vontade",
}


class MINDSVideo(Video):
    def __init__(self, filepath):
        self.filepath = filepath
        self.video_name = os.path.basename(self.filepath)

        self.capture = cv2.VideoCapture(self.filepath)
        self.video_info = VideoInfo(self.filepath)

        self.id = self._get_id()

        self.sign_id, self.word = self._get_word()

        self.sequence = self._get_sequence()

    def _get_id(self, ):
        return int(self.video_name.split('-')[0])

    def _get_sequence(self, ):
        return self.video_name.split('_')[-1][0]

    def _get_word(self):
        s = self.video_name.split('-')[1]
        s = s.split('_')[0]

        sign_id = int(s[0:2])
        word = s[2:]  # we could use the `classes` dict

        return sign_id, word


class MINDSDataset(object):
    def __init__(self, rootdir="../data/MINDS/MINDS-Libras_RGB-D"):
        self._rootdir = rootdir
        self.filepaths = self.get_files()

        self._index = 0

    def __next__(self):
        if self._index < len(self.filepaths):
            video = MINDSVideo(self.filepaths[self._index])
            self._index += 1
            return video
        else:
            raise StopIteration

    def __len__(self):
        return len(self.filepaths)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return MINDSVideo(self.filepaths[idx])

    def get_files(self):
        list_of_videos = (glob.glob(os.path.join(self._rootdir, '**/*RGB.mp4'), recursive=True))
        return list_of_videos

    def _load_gei(self, exclude=None):
        """load gei features

        Args:
            exclude (dict, optional): dict with int keys representing the person id.
            The possible correspondent values are the word to exclude in list format.
            If None, load all.
            Example:
            if exclude = {3: 'all', 4: ['Filho']}, do not load any word correspondent to
            Signalizer 03 and the word `Filho` of Signalizer 4.
            Defaults to None.
        """
        def load_geis_in_path(datapath):
            geis_paths = glob.glob(os.path.join(datapath, '*.pkl'), recursive=True)
            data = []
            for gei_path in geis_paths:
                with open(gei_path, 'rb') as f:
                    data.append(pickle.load(f))
            return data

        data = []
        label = []
        for vid_path in self.filepaths:
            video = MINDSVideo(vid_path)

            if exclude is not None:
                if video.id in exclude.keys():
                    if exclude[video.id] == 'all':
                        continue
                    elif video.word in exclude[video.id] or video.sign_id in exclude[video.id]:
                        continue

            # if video.id not in sinalizers_config or video.word not in list_of_words:
            #     continue

            # else:
            geis_path = os.path.dirname(video.filepath).replace("MINDS-Libras_RGB-D",
                                                                "MINDS-Libras_RGB-D/gei")
            geis = load_geis_in_path(geis_path)
            lbl = len(geis) * [video.sign_id]  # could be video.word

            data.extend(geis)
            label.extend(lbl)

        return data, label

    def load_features(self,
                      exclude={
                          3: 'all',
                          4: ["Filho"],
                          9: 'all'
                      },
                      dim=None,
                      crop_person=False,
                      shuffle=True,
                      flatten=True):

        X, y = self._load_gei(exclude=exclude)

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


if __name__ == "__main__":

    video = MINDSVideo(
        "/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D/Sinalizador03/01Acontecer/3-01Acontecer_1RGB.mp4"
    )

    data = MINDSDataset("/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D")

    print(data)
