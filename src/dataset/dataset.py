import glob
import os

from .video import Video


class Dataset(object):
    def __init__(self, rootdir, exts=('.avi')):
        self._rootdir = rootdir
        self._exts = exts
        self.filepaths = self.get_files()
        self._index = 0

    def __next__(self):
        if self._index < len(self.filepaths):
            video = Video(self.filepaths[self._index])
            self._index += 1
            return video
        else:
            raise StopIteration

    def __len__(self):
        return len(self.filepaths)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return Video(self.filepaths[idx])

    def get_files(self):
        list_of_videos = []
        for ext in self._exts:
            list_of_videos.extend(
                glob.glob(os.path.join(self._rootdir, f'**/*{ext}'), recursive=True))
        return list_of_videos

    def total_seconds(self):
        tot = 0
        for videopath in self.filepaths:
            video = Video(videopath)
            tot += video.get_durapytion()

        return tot

    def total_frames(self):
        tot = 0
        for videopath in self.filepaths:
            video = Video(videopath)
            tot += video.get_nb_frames()

        return tot

    def info(self):
        videos = len(self.get_files())
        seconds = self.total_seconds()
        frames = self.total_frames()

        print('Number of videos:', videos)
        print('Total frames:', frames)
        print('Total seconds:', seconds)
