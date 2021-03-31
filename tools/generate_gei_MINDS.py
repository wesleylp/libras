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
from src.dataset.MINDS import MINDSDataset
from src.utils.image import (create_context, recover_original_mask_size, select_body_parts)
from tqdm import tqdm


def run_on_video(video, model, extractor):
    # run for whole video

    annotations = dict()

    for frame_nb, frame in tqdm(enumerate(video.frame_by_frame()),
                                total=(video.get_nb_frames() - 1)):
        annotations[f'frame_{frame_nb:03d}'] = dict()

        # MINDS FRAMES ARE FULL HD, we resize it before run model
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)

        outputs = model(frame)
        data = extractor(outputs['instances'])

        iuv_arr = DensePoseResult.decode_png_data(*(data[-1].results[0]))
        i = iuv_arr[0, :, :].tolist()
        bbox = data[-1].boxes_xywh[0]

        annotations[f'frame_{frame_nb:03d}']['bbox'] = bbox
        annotations[f'frame_{frame_nb:03d}']['segm'] = i

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
        path_save_segm = path_save.replace('MINDS-Libras_RGB-D',
                                           os.path.join('MINDS-Libras_RGB-D', 'segm'))
        path_save_gei = path_save.replace('MINDS-Libras_RGB-D',
                                          os.path.join('MINDS-Libras_RGB-D', 'gei'))
        os.makedirs(path_save_segm, exist_ok=True)
        os.makedirs(path_save_gei, exist_ok=True)

        all_masks = []
        for frame, annotations in run_on_video(video, model, extractor).items():
            mask = recover_original_mask_size(annotations, (w, h))
            mask_body_parts = select_body_parts(mask, body_parts)

            all_masks.append(mask_body_parts)

        gei = np.mean(all_masks, axis=0)

        with open(os.path.join(path_save_segm, video.video_name.replace('.mp4', '.json')),
                  'w') as outfile:
            json.dump(annotations, outfile)

        with open(os.path.join(path_save_gei, video.video_name.replace('.mp4', '.pkl')), 'wb') as f:
            pickle.dump(gei, f, pickle.HIGHEST_PROTOCOL)

        cv2.imwrite(os.path.join(path_save_gei, video.video_name.replace('.mp4', '.png')),
                    (gei * 255).astype('int'))


#     return annotations


def main():

    config_file = os.path.join(
        this_filepath,
        '../src/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    model_url = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

    dataset_path_root = os.path.join(this_filepath, '../data/MINDS-Libras_RGB-D')
    ufop_dataset = MINDSDataset(dataset_path_root)

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
