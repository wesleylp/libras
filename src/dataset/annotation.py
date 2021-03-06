import json
import os
import numpy as np


class Annotation(object):
    def __init__(self, filepath):
        self._filepath = filepath
        self._video_segmentation = None

        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                self._video_segmentation = json.load(f)
        else:
            raise ValueError("Not a valid file: ", self._filepath)

    def get_frame_annotation(self, frame):
        bbox = np.array(self._video_segmentation[f"frame_{frame:03d}"]['bbox'])
        segm = np.array(self._video_segmentation[f"frame_{frame:03d}"]['segm'])

        return {'bbox': bbox, 'segm': segm}

    def get_video_annotation(self):
        video_annot = {}

        for frame, annots in self._video_segmentation.items():
            video_annot[frame] = dict()
            video_annot[frame]['bbox'] = np.array(annots['bbox'])
            video_annot[frame]['segm'] = np.array(annots['segm'])

        return video_annot
