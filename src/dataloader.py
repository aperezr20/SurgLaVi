import os
import re
import sqlite3

import torch
from torch.utils.data import Dataset


# Dataset level identifiers for naming
DATASET_IDENTIFIER = {
    1: 'task',
    2: 'step', 
    3: 'phase'
}


class SurgLaViDataset(Dataset):
    """
    Simple data loader for reading video clips from an SQLite database.
    
    Loads video annotations with timestamps and captions, returning random tensors
    as placeholders for actual video data.
    """
    
    def __init__(
        self,
        db_path,
        video_root,
        transform=None,
        num_frames = 4,
        filter_pairs = True,
        enhanced_captions = True,
        text_preprocess = True, 
        level_ids = [1, 2, 3],
        min_duration = 0,
        max_duration = 500,
    ):
        """
        Initialize the video data loader.
        
        Args:
            ann_file: Tuple of (database_path, video_root, level_ids)
            transform: Transform function to apply to video frames
            num_frames: Number of frames to sample from each video clip
            level_ids: List of hierarchy levels to include (1=task, 2=step, 3=phase)
        """
        super().__init__()
        
        self.num_frames = num_frames
        self.transform = transform
        self.level_ids = level_ids
        self.database_path = db_path
        self.video_root = video_root
        
        # Connect to database
        self.connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro", 
            uri=True,
            check_same_thread=False
        )
        self.cursor = self.connection.cursor()
        
        # Filter parameters
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.filter_pairs = filter_pairs
        self.enhanced_captions = enhanced_captions
        self.level_ids_str = ', '.join(map(str, self.level_ids))
        self.text_preprocess = text_preprocess

        # Load annotations
        self.annotations = self._load_annotations()
        self.num_examples = len(self.annotations)
        
        # Create dataset name
        self.name = '-'.join(DATASET_IDENTIFIER[i] for i in self.level_ids)
        
        print(f"Loaded {self.num_examples} annotations for dataset: {self.name}")
    
    def _load_annotations(self):
        """Load all annotations from the database."""
        
        # Choose caption column
        caption_column = 'caption' if self.enhanced_captions else 'raw_narration'
        
        # Build WHERE conditions
        conditions = [
            f"duration <= {self.max_duration} AND duration > {self.min_duration}",
            f"level_id IN ({self.level_ids_str})",
        ]
        
        # Add content filters if enabled
        if self.filter_pairs:
            conditions.append("is_descriptive=1 AND is_surgical_content=1")
        
        # Build query
        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT caption_id, video_id, level_id, start_time, end_time, {caption_column}
            FROM captions 
            WHERE {where_clause}
            ORDER BY caption_id
        """
        
        # Execute query
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Process results
        annotations = []
        for row in results:
            caption_id, video_id, level_id, start_time, end_time, caption = row
            
            # Get video info
            video_query = "SELECT name, fps FROM videos WHERE video_id = ?;"
            self.cursor.execute(video_query, (video_id,))
            video_result = self.cursor.fetchone()
            
            if video_result is None:
                continue
                
            name, fps = video_result
            video_path = os.path.join(self.video_root, name, f"{name}.mp4")
            
            # Clean caption
            if self.text_preprocess:
                caption = self._pre_text(caption.replace('The surgeon', ''))
            
            annotations.append({
                "video_path": video_path,
                "caption": caption,
                "timestamps": (start_time, end_time),
                "fps": fps,
                "video_name": name,
                "level_id": level_id
            })
        
        return annotations
    

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        
        # TODO: Replace with actual video loading
        video_frames = self._load_video_clip(annotation)
        
        return video_frames, annotation['caption'], index
    
    def _load_video_clip(self, annotation):
        """
        TODO: Load and transform a video clip.
        Currently returns a random tensor placeholder.
        """
        fps = annotation['fps']
        start, end = annotation['timestamps']
        video_path = annotation['video_path']
        return torch.rand(self.num_frames, 3, 224, 224)
    
    def get_stats(self):
        """Get dataset statistics."""
        if not self.annotations:
            return {}
        
        level_counts = {}
        durations = []
        
        for ann in self.annotations:
            level_id = ann['level_id']
            level_counts[level_id] = level_counts.get(level_id, 0) + 1
            
            start, end = ann['timestamps']
            durations.append(end - start)
        
        return {
            'total_samples': len(self.annotations),
            'level_distribution': level_counts,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations)
        }

    def _pre_text(self, text, max_l=None):
        """Pre-process text"""
        text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
        # We replace "The surgeon" as a lot of enhanced captions begin with this 
        text = text.replace('-', ' ').replace('/', ' ').replace('The surgeon', '') 
        text = re.sub(r"\s{2,}", ' ', text)
        text = text.rstrip('\n').strip(' ')
        return text
