import cv2
import numpy as np

# for body parts see:
# https://github.com/facebookresearch/DensePose/blob/master/challenge/2019_COCO_DensePose/data_format.md
# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
body_parts = {
    "Torso": [1, 2],
    "RightHand": [3],
    "LeftHand": [4],
    "UpperArmLeft": [15, 17],
    "UpperArmRight": [16, 18],
    "LowerArmLeft": [19, 21],
    "LowerArmRight": [20, 22],
    "Head": [23, 24]
}

EPS = 1e-16


def select_body_parts(mask, list_of_body_parts):
    """make a mask considering only the list_of_body_parts

    Args:
        mask (numpy.array): body parts segmentation
        list_of_body_parts (list of str): body parts desired

    Returns:
        [numpy.array]: binary mask where 1 denotes the body parts selected
    """

    new_mask = np.zeros(mask.shape).astype(np.bool)

    for body_part in list_of_body_parts:
        idxs = body_parts[body_part]

        for idx in idxs:
            m_ = (mask == idx)
            new_mask = np.bitwise_or(new_mask, m_)

    return new_mask.astype(np.uint8)


def overlay_segmentation(image_bgr,
                         segmentation,
                         colormap=cv2.COLORMAP_PARULA,
                         alpha=0.6,
                         inplace=True):
    """Overlay segmentation on image

    Args:
        image_BGR (numpy.array): image in BGR
        segmentation (dict): Dictionay with 'bbox' and 'segm'
    """
    if inplace:
        image_target_bgr = image_bgr
    else:
        image_target_bgr = image_bgr * 0

    bbox_xywh = segmentation['bbox']
    x, y, w, h = [int(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return image_bgr

    segm = segmentation['segm']
    levels = 255.0 / segm.max()
    mask_bg = np.tile((segm == 0)[:, :, np.newaxis], [1, 1, 3])

    segm = segm.astype(np.float32) * levels
    segm = segm.clip(0, 255).astype(np.uint8)
    segm = cv2.applyColorMap(segm, cv2.COLORMAP_PARULA)

    segm[mask_bg] = image_target_bgr[y:y + h, x:x + w, :][mask_bg]

    # img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # segm_hsv = cv2.cvtColor(segm, cv2.COLOR_BGR2HSV)

    image_target_bgr[y:y + h, x:x + w, :] = (image_target_bgr[y:y + h, x:x + w, :] * (1.0 - alpha) +
                                             segm * alpha)

    return image_target_bgr.astype(np.uint8)


def recover_original_mask_size(annot, orig_w_h):
    """Generate segmentation image of the same size of the original one.

    Args:
        annot (dict): Dictionaty containing "bbox" and "segm"
        orig_w_h (tuple): (width, heigth) of original image

    Returns:
        numpy.array: image body parts segmentation.
    """
    img = np.zeros(orig_w_h).T

    bbox_xywh = annot['bbox']
    segm = annot['segm']
    x, y, w, h = [int(v) for v in bbox_xywh]

    img[y:y + h, x:x + w] = segm
    return img


def generate_gei(video,
                 output_dim='same',
                 normalized=True,
                 body_parts=[
                     'RightHand',
                     'LeftHand',
                     'UpperArmLeft',
                     'UpperArmRight',
                     'LowerArmLeft',
                     'LowerArmRight',
                     'Head',
                 ]):
    """generate gei

    Args:
        video (object of class Video): Input video to compute gei
        output_dim (str or tuple, optional): if 'same' output dim is equal of input dimensions.
        In case of tuple (w,h), resize image to (w,h)
        normalized (bool, optional): If True normalize GEI to be in range (0,1)
        body_parts (list of str, optional): List of body parts desired to compute gei
    """
    all_videos_mask = []

    video_annot = video.segmentation.get_video_annotation()
    w, h = video.get_width_height()

    for frame, annotations in video_annot.items():
        mask = recover_original_mask_size(annotations, (w, h))
        mask_body_parts = select_body_parts(mask, body_parts)

        if output_dim != 'same':
            mask_body_parts = cv2.resize(mask_body_parts, output_dim, interpolation=cv2.INTER_CUBIC)
            mask_body_parts = mask_body_parts - mask_body_parts.flatten().min()
            mask_body_parts = mask_body_parts / (mask_body_parts.flatten().max() + EPS)
            mask_body_parts = mask_body_parts.clip(0., 1.)

        all_videos_mask.append(mask_body_parts)

    gei = np.mean(all_videos_mask, axis=0)

    if normalized:
        gei = gei - gei.flatten().min()
        gei = gei / (gei.flatten().max() + EPS)
        gei = gei.clip(0., 1.)

    return gei


def crop_person_img(img):

    people_in = np.where(img > 0)
    h_min = people_in[0].min()
    h_max = people_in[0].max()
    w_min = people_in[1].min()
    w_max = people_in[1].max()

    return img[h_min:h_max + 1, w_min:w_max + 1].copy()
