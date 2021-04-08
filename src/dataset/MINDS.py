import glob
import os
import pickle
import sys

import cv2
import numpy as np
import sklearn

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../detectron2/projects/DensePose/'))

from src.utils.image import crop_person_img

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

    def _load_gei(
        self,
        exclude=None,
        dim=None,
        crop_person=False,
    ):
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
        def load_gei_in_path(datapath):
            with open(datapath, 'rb') as f:
                data = pickle.load(f)
            return data

        data = []
        label = []
        for vid_path in self.filepaths:
            video = MINDSVideo(vid_path)

            if exclude is not None:
                if video.id in exclude.keys():
                    if exclude[video.id] == 'all' or video.word in exclude[
                            video.id] or video.sign_id in exclude[video.id]:
                        continue

            gei_path = video.filepath.replace("MINDS-Libras_RGB-D", "MINDS-Libras_RGB-D/gei")
            gei = load_gei_in_path(gei_path.replace('.mp4', '.pkl'))

            # we perform crop and reescale here because the original images are 1920x1080
            # it saves memory
            if crop_person:
                gei = crop_person_img(gei).astype('float64')

            if dim is not None:
                gei = cv2.resize(gei, dim, interpolation=cv2.INTER_CUBIC).astype('float64')

            data.append(gei)
            label.append(video.sign_id)  # could be video.word

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

        X, y = self._load_gei(exclude=exclude, dim=dim, crop_person=crop_person)

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y, random_state=0)

        if flatten:
            X = [x.flatten().astype('float64') for x in X]

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == "__main__":

    import time

    video = MINDSVideo(
        "/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D/Sinalizador03/01Acontecer/3-01Acontecer_1RGB.mp4"
    )

    minds_dataset = MINDSDataset("/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D")

    st = time.time()
    X, y = minds_dataset.load_features(dim=(64, 48), crop_person=True, shuffle=True, flatten=True)
    print(f"elapsed time to load data: {time.time() - st} seconds")

    print(minds_dataset)
