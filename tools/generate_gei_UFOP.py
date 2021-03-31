import json
import os
import pickle
import sys

import cv2
import numpy as np

this_filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_filepath, '../src/detectron2/projects/DensePose/'))

from densepose import add_densepose_config, add_hrnet_config
from densepose.data.structures import DensePoseResult
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose import (DensePoseResultsContourVisualizer,
                                     DensePoseResultsFineSegmentationVisualizer,
                                     DensePoseResultsUVisualizer, DensePoseResultsVVisualizer)
from densepose.vis.extractor import CompoundExtractor, create_extractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from src.dataset.UFOP import UFOPDataset
from src.utils.image import (create_context, recover_original_mask_size, select_body_parts)
from tqdm import tqdm

# IN THE UFOP-DATASET WE HAVE TO BE A LITTLE MORE CAREFULL
# AS THE GESTURE IS PERFORMED MORE THAN ONCE IN EACH VIDEO


def run_on_video(video, model, extractor):
    # run for whole video

    frame_nb = 0
    interval_nb = 0
    frames_with_gestures = video.get_movement_frames()

    annotations = dict()

    # for interval in frames_with_gestures:

    interval = frames_with_gestures[interval_nb]
    init_interval, end_interval = interval[0], interval[1]
    annotations[f'{init_interval:03d}_{end_interval:03d}'] = dict()

    # video.reset()

    for frame_nb, frame in tqdm(enumerate(video.frame_by_frame()),
                                total=(video.get_nb_frames() - 1)):

        if frame_nb >= init_interval and frame_nb <= end_interval:
            annotations[f'{init_interval:03d}_{end_interval:03d}'][f'frame_{frame_nb:03d}'] = dict()

            outputs = model(frame)
            data = extractor(outputs['instances'])

            iuv_arr = DensePoseResult.decode_png_data(*(data[-1].results[0]))
            i = iuv_arr[0, :, :].tolist()
            bbox = data[-1].boxes_xywh[0]

            annotations[f'{init_interval:03d}_{end_interval:03d}'][f'frame_{frame_nb:03d}'][
                'bbox'] = bbox
            annotations[f'{init_interval:03d}_{end_interval:03d}'][f'frame_{frame_nb:03d}'][
                'segm'] = i

            if frame_nb >= end_interval and interval_nb < len(frames_with_gestures) - 1:
                interval_nb += 1
                interval = frames_with_gestures[interval_nb]
                init_interval, end_interval = interval[0], interval[1]
                annotations[f'{init_interval:03d}_{end_interval:03d}'] = dict()

    return annotations


def run_on_dataset(dataset,
                   model,
                   extractor,
                   body_parts=[
                       'RightHand',
                       'LeftHand',
                       'UpperArmLeft',
                       'UpperArmRight',
                       'LowerArmLeft',
                       'LowerArmRight',
                       'Head',
                   ]):

    for video in tqdm(dataset, total=len(dataset)):

        w, h = video.get_width_height()

        path_save = os.path.dirname(video.filepath)
        path_save_segm = path_save.replace('LIBRAS-UFOP', os.path.join('LIBRAS-UFOP', 'segm'))
        path_save_gei = path_save.replace('LIBRAS-UFOP', os.path.join('LIBRAS-UFOP', 'gei'))
        os.makedirs(path_save_segm, exist_ok=True)
        os.makedirs(path_save_gei, exist_ok=True)

        clips = run_on_video(video, model, extractor)

        for clip, segmentation in clips.items():

            # for segmentation in clips:
            all_clip_masks = []

            for frame, annotations in segmentation.items():
                mask = recover_original_mask_size(annotations, (w, h))
                mask_body_parts = select_body_parts(mask, body_parts)

                all_clip_masks.append(mask_body_parts)

            gei = np.mean(all_clip_masks, axis=0)

            with open(os.path.join(path_save_segm, f'segm_{clip}.json'), 'w') as outfile:
                json.dump(segmentation, outfile)

            with open(os.path.join(path_save_gei, f'gei_{clip}.pkl'), 'wb') as f:
                pickle.dump(gei, f, pickle.HIGHEST_PROTOCOL)

            cv2.imwrite(os.path.join(path_save_gei, f'gei_{clip}.png'), (gei * 255).astype('int'))


#     return annotations


def main():

    config_file = os.path.join(
        this_filepath,
        '../src/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    model_url = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

    dataset_path_root = os.path.join(this_filepath, '../data/LIBRAS-UFOP')
    ufop_dataset = UFOPDataset(dataset_path_root, os.path.join(dataset_path_root, 'labels.txt'))

    # Inference with a keypoint detection model
    cfg = get_cfg()

    add_densepose_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_url

    predictor = DefaultPredictor(cfg)
    context = create_context((
        'bbox',
        'dp_segm',
    ))

    # visualizer = context["visualizer"]
    extractor = context["extractor"]

    run_on_dataset(ufop_dataset, predictor, extractor)


if __name__ == "__main__":
    main()
