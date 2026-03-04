import logging
import os
import numpy as np
import torch
import re

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.utils import (
    load_image_lists,
    load_coco_annotations,
    get_keyframe_data,
    get_sequence,
    read_frame_sequence
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WorkflowDataset(Dataset):
    media_type = "video"

    def __init__(self, config, ann_file, transform, sample_type='middle',num_frames=16):
        super().__init__()
        
        (
            self.label_file, 
            self.data_root, 
            self.media_type,
            self.frame_lists,
            self.sample_rate,
            self.zero_fill,
            self.image_type,
            self.name,
            self.task
        ) = ann_file
        
        self.transform = transform
        self.num_frames = num_frames
        self._seq_len = self.num_frames * self.sample_rate

        labels, self.categories, self.prompts = load_coco_annotations(
            self.label_file, self.task
        )
        
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = load_image_lists(self.frame_lists, self.data_root, labels)

        assert len(labels) == len(self._image_paths)

        self.labels = [
            labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        (
            self._keyframe_indices,
            _,
        ) = get_keyframe_data(self.labels, self.keyframe_mapping)
        
        self.num_examples = len(self._keyframe_indices)

    def keyframe_mapping(self, video_idx, label_idx, frame_num):
        if self.name == 'sarrarp50':
            return label_idx * 5
        else:
            return label_idx

    def __len__(self):
        return self.num_examples

    def format_frame_path(self, video_name, frame_num):
        if any(keyword in self.name for keyword in ['cholec80', 'bypass']):
            video_name = '{}/{}_{}.{}'.format(
                video_name, 
                video_name, 
                str(frame_num).zfill(self.zero_fill), 
                self.image_type
                )
        elif self.name == 'sarrarp50':
            video_name = '{}/rgb/{}.{}'.format(
                video_name, 
                str(frame_num).zfill(self.zero_fill), 
                self.image_type
                )
        elif any(keyword in self.name for keyword in ['grasp', 'autolaparo', 'cholect50', 'heichole']):
            video_name = '{}/{}.{}'.format(
                video_name, 
                str(frame_num).zfill(self.zero_fill), 
                self.image_type
                )
        return video_name

    def resize_frames(self, frames, target_size=224):
        resize_transform = transforms.Resize(
                (target_size, target_size),
                interpolation=InterpolationMode.BICUBIC,
            )
        resized_frames = [resize_transform(frame) for frame in frames]
        return resized_frames

    def _get_item_core(self, idx, resize=False):
        video_idx, label_idx, frame_num, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        video_num = int(re.findall(r'\d+', video_name)[0])
        image_path = self.format_frame_path(video_name, frame_num)
        image_path = os.path.join(self.data_root, image_path)
        found_idx = self._image_paths[video_idx].index(image_path)

        label = self.labels[video_idx][frame_num]

        if self.task in ['phases', 'steps']:
            label = label[0]
        else:
            binary_label = np.zeros(len(self.categories))
            for i in label:
                binary_label[i - 1] = 1
            label = binary_label

        seq = get_sequence(
            found_idx,
            self._seq_len // 2,
            self.sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self.num_frames,
            online=False,
        )

        frame_paths = [self._image_paths[video_idx][frame] for frame in seq]
        clip = read_frame_sequence(frame_paths)

        if resize:
            clip = self.resize_frames(clip)

        clip = self.transform(torch.stack(clip))
        return clip, label, video_num, int(frame_num)

    def __getitem__(self, idx):
        try:
            return self._get_item_core(idx, resize=False)
        except Exception as e:
            print(e)


class CholecT50Dataset(WorkflowDataset):
    def __init__(self, config, ann_file, transform, num_frames=1, sample_type='middle', additional_param=None, **kwargs):
        super().__init__(config, ann_file, transform, num_frames, **kwargs)

    def __getitem__(self, idx):
        try:
            video_idx, label_idx, frame_num, center_idx = self._keyframe_indices[idx]
            video_name = self._video_idx_to_name[video_idx]
            video_num = int(re.findall(r'\d+', video_name)[0])
            image_path = self.format_frame_path(video_name, frame_num)
            
            image_path = os.path.join(self.data_root, image_path)
            found_idx = self._image_paths[video_idx].index(image_path)
            label = self.labels[video_idx][frame_num]
            
            assert len(label) == 1

            label = label[0]
            seq = get_sequence(
            found_idx,
            self._seq_len // 2,
            self.sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self.num_frames, 
            online = False,
            )

            frame_paths = [self._image_paths[video_idx][frame] for frame in seq]
            clip = read_frame_sequence(frame_paths)
            clip = self.resize_frames(
                clip
            )
            
            clip = self.transform(torch.stack(clip))

            return clip, label, video_num, int(frame_num)

        except Exception as e:
            print(e)