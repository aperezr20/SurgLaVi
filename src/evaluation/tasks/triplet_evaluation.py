import numpy as np
import os
import re
import pandas as pd
import logging
import ivtmetrics
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from tasks.shared_utils import convert_to_2d_array
import torch

logger = logging.getLogger(__name__)

class TripletEvaluator:
    """
    Class for evaluating surgical phase recognition metrics from prediction dataframes
    """
    
    def __init__(self, tasks, prefix=""):
        """Initialize the evaluator with optional phase names"""
        self.tasks = tasks
        self.prefix = prefix
        
        self.metrics = {
            'triplet': ivtmetrics.Recognition(100),
            'instrument': ivtmetrics.Recognition(6),
            'verb': ivtmetrics.Recognition(10),
            'target': ivtmetrics.Recognition(15)
        }
        
        for name, metric in self.metrics.items():
            metric.reset_global()
            metric.reset()


    def evaluate(self, predictions_df, group_by_video=True):
        """
        Evaluate metrics from a predictions dataframe
        
        Args:
            predictions_df: DataFrame with columns 'video_idx', 'frame_idx', 'prediction', 'ground_truth'
            group_by_video: Whether to evaluate metrics per video
            
        Returns:
            Dictionary of metrics with prefix as the main key
        """
        
        # Initialize empty results dictionary with hierarchical structure
        results = {
            self.prefix: {
                'triplet': {},
                'instrument': {},
                'verb': {},
                'target': {},
            }
        }
        
        for task, df in predictions_df.items():
            video_groups = df.groupby('video_idx')

            for video_idx, video_df in video_groups:
                preds = video_df['prediction'].values
                preds = np.stack(convert_to_2d_array(preds))
                labels = np.stack(video_df['ground_truth'].values)
                self.metrics[task].update(labels, preds)
                self.metrics[task].video_end()



        for task, metric in self.metrics.items():
            values = metric.compute_video_AP(ignore_null=False)
            results[self.prefix][task] = values
        
        results[self.prefix]['instrument-target'] = self.metrics['triplet'].compute_video_AP(
            'it', 
            ignore_null=False
            )

        results[self.prefix]['instrument-verb'] = self.metrics['triplet'].compute_video_AP(
            'iv', 
            ignore_null=False
            )
        
        return results