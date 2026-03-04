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

class WorkflowEvaluator:
    """
    Class for evaluating surgical phase recognition metrics from prediction dataframes
    """
    
    def __init__(self, phases, prefix=""):
        """Initialize the evaluator with optional phase names"""
        self.phases = phases
        self.prefix = prefix
        self.num_phases = len(self.phases)
        
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
                'accuracy': [],
                'f1_average': [],
                'f1_per_class': [],
                'video_ids': []
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
                preds = convert_to_2d_array(preds)
                preds = preds.argmax(axis=1)

                labels = video_df['ground_truth'].values
                
                # Calculate metrics
                acc = self.calc_accuracy(preds, labels)
                
                f1_class, f1_average = self.calc_f1(preds, labels)

                
                # Store results
                results[self.prefix]['accuracy'].append(acc)
                results[self.prefix]['f1_average'].append(f1_average)
                results[self.prefix]['f1_per_class'].append(f1_class)
                results[self.prefix]['video_ids'].append(video_idx)
                
                logger.info(f'Video #{video_idx} Accuracy: {acc:.4f}')
                logger.info(f'Video #{video_idx} F1 average: {f1_average:.4f}')
        
        # Calculate overall metrics
        all_preds = np.stack(predictions_df['prediction'].values)
        all_labels = predictions_df['ground_truth'].values
        
        all_preds = all_preds.argmax(axis=1)

        overall_acc = self.calc_accuracy(all_preds, all_labels)
        overall_f1_class, overall_f1_average = self.calc_f1(all_preds, all_labels)
        # Add overall metrics to results
        results[self.prefix]['overall_accuracy'] = overall_acc
        results[self.prefix]['overall_f1_average'] = overall_f1_average
        results[self.prefix]['overall_f1_per_class'] = overall_f1_class
        
        # Calculate averages across videos
        results[self.prefix]['video_avg_accuracy'] = np.mean(results[self.prefix]['accuracy'])
        results[self.prefix]['video_avg_f1'] = np.mean(results[self.prefix]['f1_average'])
        results[self.prefix]['video_avg_f1_class'] = np.array(results[self.prefix]['f1_per_class']).mean(axis=0)
        
        # Logger summary
        logger.info('=' * 50)
        logger.info(f'Overall Accuracy: {overall_acc:.4f}')
        logger.info(f'Overall F1 average: {overall_f1_average:.4f}')
        logger.info('-' * 50)
        logger.info(f'Video-wise average Accuracy: {results[self.prefix]["video_avg_accuracy"]:.4f}')
        logger.info(f'Video-wise average F1: {results[self.prefix]["video_avg_f1"]:.4f}')
        logger.info(f'Video-wise average F1 classes: {np.array2string(f1_class, precision=4)}')
        
        for idx, phase_name in enumerate(self.phases):
            logger.info(f'F1 {phase_name}: {np.array2string(results[self.prefix]["video_avg_f1_class"][idx], precision=4)}')
        
        return results
