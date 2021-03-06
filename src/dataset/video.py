import os
import subprocess
import shlex
import json

import cv2
from .annotation import Annotation


class Video(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.video_name = os.path.basename(self.filepath)
        self.capture = cv2.VideoCapture(self.filepath)
        self.video_info = VideoInfo(self.filepath)
        self.segmentation = Annotation(self.filepath.replace(".avi", ".json"))

        s = self.video_name.split('_')
        self.id = s[2].split('.')[0]
        self.word = s[0]
        self.background = s[1]

    def reset(self):
        ret = self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return ret

    def get_frame(self, frame_nb):
        ret = self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        if not ret:
            print('fail setting {:d} frame number!'.format(frame_nb))

        # read video
        ret, frame = self.capture.read()

        if not ret:
            print('fail reading {:d} frame number!'.format(frame_nb))

        return frame

    def get_nb_frames(self):
        return self.video_info.get_nb_frames()

    def get_fps(self):
        return self.video_info.get_fps()

    def get_duration(self):
        return self.video_info.get_duration()

    def get_width_height(self):
        return self.video_info.get_width_height()

    def info(self):
        seconds = self.get_duration()
        frames = self.get_nb_frames()
        fps = self.get_fps()
        w, h = self.get_width_height()

        print('filepath:', self.filepath)
        print('Total frames:', frames)
        print('Total seconds:', seconds)
        print('fps:', fps)
        print('width, height:', w, h)

    def frame_by_frame(self):
        while self.capture.isOpened():
            success, frame = self.capture.read()
            if success:
                yield frame
            else:
                break


class VideoInfo(object):
    def __init__(self, videopath):
        try:
            with open(os.devnull, 'w') as tempf:
                subprocess.check_call(["ffprobe", "-h"], stdout=tempf, stderr=tempf)
        except IOError:
            raise IOError('ffprobe not found!')

        if os.path.isfile(videopath):
            cmd = "ffprobe -v error -print_format json -show_streams -show_format"

            # makes a list with contents of cmd
            args = shlex.split(cmd)
            # append de video file to the list
            args.append(videopath)

            # Running ffprobe process and loads it in a json structure
            ffoutput = subprocess.check_output(args).decode('utf-8')
            ffoutput = json.loads(ffoutput)

            # Check available information on the file
            for i in range(len(ffoutput['streams'])):
                if ffoutput['streams'][i]['codec_type'] == 'video':
                    self._video_idx = i
                elif ffoutput['streams'][i]['codec_type'] == 'audio':
                    self._audio_idx = i
                elif ffoutput['streams'][i]['codec_type'] == 'subtitle':
                    self._subtitle_idx = i

            self._ffoutput = ffoutput

        else:
            raise ValueError("Not a valid file: ", videopath)

    def get_filepath(self):
        return self._ffoutput['format']['filename']

    def get_nb_frames(self):
        return int(self._ffoutput['streams'][self._video_idx]['nb_frames'])

    def get_duration(self):
        return float(self._ffoutput['streams'][self._video_idx]['duration'])

    def get_width_height(self):
        w = int(self._ffoutput['streams'][self._video_idx]['width'])
        h = int(self._ffoutput['streams'][self._video_idx]['height'])
        return (w, h)

    def get_fps(self):
        fps = self._ffoutput['streams'][self._video_idx]['r_frame_rate']

        if fps is not None:
            idx = fps.find('/')
            if idx == -1:
                return None
            num = float(fps[:idx])
            den = float(fps[idx + 1:])
        return num / den
