import logging
import torch
import os
import json
import re
import numpy as np
from os.path import join
from tqdm import trange
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import PILToTensor
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def load_image_from_path(image_path):
    image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image


def load_image_lists(file, data_root, labels):
    """
    Loading image paths from corresponding files.

    Args:
        file (str): 
                csv file path where frame lists are located. 
                Each row follows: original_vido_id video_id frame_id path

        data_root (str): path where extracted frames are located

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """

    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    split_videos = list(labels.keys())

    with open(file, "r") as f:
        for line in f:
            row = line.split()
            # The format of each row should follow:
            # original_vido_id video_id frame_id path
            assert len(row) == 4
            video_name = row[0]
            if video_name not in split_videos:
                continue

            if video_name not in video_name_to_idx:
                idx = len(video_name_to_idx)
                video_name_to_idx[video_name] = idx
                video_idx_to_name.append(video_name)

            data_key = video_name_to_idx[video_name]
            image_paths[data_key].append(os.path.join(data_root, row[3]))

    image_paths = [image_paths[i] for i in range(len(image_paths))]
    logger.info(f"Finished loading image paths from: {file}")

    return image_paths, video_idx_to_name


def load_coco_annotations(filename, task):
    """
    Loading boxes and labels from coco json files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            {`bbox`:<box_coord>, *`box_labels`:<label> where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    labels, count, categories, prompts = parse_annotations(filename, task)
    
    logger.info(f"Finished loading annotations from: {filename}")
    logger.info("Number of annotations: %d" % count)


    return labels, categories, prompts


def parse_annotations(filename, task):
    """
    Parse bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of bounding boxes coco annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    count = 0
    annotated_frames_dict = {}
    id2frame = {}
    
    with open(filename, "r") as f:
        data = json.load(f)
    
    if task == 'triplet':
        instrument = data['instrument_categories']
        verb = data['verb_categories']
        target = data['target_categories']
        triplet = data['triplet_categories']
        categories = [triplet, instrument, verb, target]
        prompts = {}
        for task_cat in categories:
            prompts[task_cat[0]['supercategory']] = [i['description'] for i in task_cat]
    
    elif task == 'instruments':
        categories = [i['name'] for i in data[f'categories']]
        prompts = [i['description'] for i in data[f'categories']]
    else:
        categories = [i['name'] for i in data[f'{task}_categories']]
        prompts = [i['description'] for i in data[f'{task}_categories']]
        
    
    for image in data['images']:
        id2frame[image['id']] = (image['video_name'], image['frame_num'], image['width'], image['height'])        
        if image['video_name'] not in annotated_frames_dict:
            annotated_frames_dict[image['video_name']] = {image['frame_num']:[]}

    for annotation in data['annotations']:
        video_name, frame_num, width, height = id2frame[annotation['image_id']]
        label = annotation[task]

        if frame_num not in annotated_frames_dict[video_name]:
            annotated_frames_dict[video_name][frame_num] = []

        if isinstance(label, list):
            annotated_frames_dict[video_name][frame_num].extend(label)
        else:
            annotated_frames_dict[video_name][frame_num].append(label)
        
        count += 1

    return annotated_frames_dict, count, categories, prompts


def get_keyframe_data(labels, keyframe_mapping):
    """
    Getting keyframe indices in the dataset.
    """
    keyframe_indices = []
    keyframe_labels = []
    count = 0
    for video_idx in range(len(labels)):
        keyframe_labels.append([])
        for sec_idx, sec in enumerate(labels[video_idx]):
            keyframe_indices.append((video_idx, sec_idx, sec, keyframe_mapping(video_idx, sec_idx, sec)))
            keyframe_labels[video_idx].append(
                labels[video_idx][sec]
            )
            count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_labels


def get_sequence(center_idx, half_len, sample_rate, num_frames, length, online=False):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    if not online:
        if length % 2 == 0:
            seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))
        else:
            seq = list(range(center_idx - half_len, center_idx + half_len + 1, sample_rate))
    
    else:
        seq = list(range(center_idx, center_idx - half_len * 2, -sample_rate))
        seq.sort()
 

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def load_image(full_path):
    return load_image_from_path(full_path)[0]

def read_frame_sequence(frame_paths):
    with ThreadPoolExecutor() as executor:
        clip = list(executor.map(load_image, frame_paths))
    return clip