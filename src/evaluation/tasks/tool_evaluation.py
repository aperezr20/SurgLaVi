import numpy as np
import os
import re
import pandas as pd
import logging
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from tasks.shared_utils import convert_to_2d_array
import torch

logger = logging.getLogger(__name__)

class ToolPresenceEvaluator:
    """
    Class for evaluating surgical phase recognition metrics from prediction dataframes
    """
    
    def __init__(self, tools, prefix=""):
        """Initialize the evaluator with optional phase names"""
        self.tools = tools
        self.prefix = prefix
        self.num_classes = len(self.tools)
        
    def calc_accuracy(self, preds, labels):
        """Calculate simple accuracy"""
        correct = (np.array(preds) == np.array(labels)).astype(int).sum().item()
        total = len(labels)
        return correct / total
    
    def calc_f1(self, preds, labels):
        """Calculate F1 score for each class and average"""
        fixed_labels = list(range(self.num_phases))
        f1_class = f1_score(labels, preds, average=None, labels=fixed_labels)
        f1_average = f1_score(labels, preds, average='macro', labels=fixed_labels)
        return f1_class, f1_average

    def calc_mAP(self, preds, labels):
        """Calculate AP score for each class and get mAP"""
        precision = {}
        recall = {}
        threshs = {}
        ap = {}

        for c in range(0, self.num_classes):
            precision[c], recall[c], threshs[c] = precision_recall_curve(labels[:, c], preds[:, c])
            ap[c] = average_precision_score(labels[:, c], preds[:, c])
        
        mAP = np.nanmean(list(ap.values()))
        ap = list(ap.values())
        return ap, mAP

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
                'mAP': [],
                'AP_per_class': [],
            }
        }
        
        if group_by_video:
            # Group predictions by video
            video_groups = predictions_df.groupby('video_idx')
            
            for video_idx, video_df in video_groups:
                # Sort by frame index
                video_df = video_df.sort_values('frame_idx')

                
                # Extract predictions and ground truth
                preds = video_df['prediction'].values
                preds = np.stack(convert_to_2d_array(preds))
                labels = np.stack(video_df['ground_truth'].values)

                # Calculate metrics
                ap, mAP = self.calc_mAP(preds, labels)

                # Store results
                results[self.prefix]['mAP'].append(mAP)
                results[self.prefix]['AP_per_class'].append(ap)
                logger.info(f'Video #{video_idx} mAP: {mAP:.4f}')

        
        # Calculate overall metrics
        all_preds = np.stack(convert_to_2d_array(predictions_df['prediction'].values))
        all_labels = np.stack(predictions_df['ground_truth'].values)
        overall_ap, overall_mAP = self.calc_mAP(all_preds, all_labels)
        results[self.prefix]['overall_mAP'] = overall_mAP
        results[self.prefix]['overall_AP_per_class'] = overall_ap
        
        # Calculate averages across videos
        results[self.prefix]['video_avg_mAP'] = np.mean(results[self.prefix]['mAP'])
        results[self.prefix]['video_avg_AP_class'] = np.array(results[self.prefix]['AP_per_class']).mean(axis=0)
        
        # Logger summary
        logger.info('=' * 50)
        logger.info(f'Overall mAP: {overall_mAP:.4f}')
        logger.info('-' * 50)
        logger.info(f'Video-wise mAP: {results[self.prefix]["video_avg_mAP"]:.4f}')
        logger.info(f'Video-wise average AP classes: {np.array2string(np.array(overall_ap), precision=4)}')

        for idx, tool_name in enumerate(self.tools):
            logger.info(f'mAP {tool_name}: {np.array2string(results[self.prefix]["video_avg_AP_class"][idx], precision=4)}')
        
        return results
