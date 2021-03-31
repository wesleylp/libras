import glob
import os
import re

import cv2

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

        self.id = int(self.video_name.split('-')[0])

        self.sequence = self.video_name.split('_')[-1][0]


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


if __name__ == "__main__":

    video = MINDSVideo(
        "/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D/Sinalizador03/01Acontecer/3-01Acontecer_1RGB.mp4"
    )

    data = MINDSDataset("/home/wesley.passos/repos/libras/data/MINDS-Libras_RGB-D")

    print(data)
