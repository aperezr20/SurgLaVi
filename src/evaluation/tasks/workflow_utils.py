import datetime
import logging
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import ivtmetrics
import torch.nn as nn
import surgclip
from einops import rearrange
from utils.basic_utils import MetricLogger
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from tasks.workflow_evaluation import WorkflowEvaluator
from tasks.tool_evaluation import ToolPresenceEvaluator
from tasks.triplet_evaluation import TripletEvaluator

logger = logging.getLogger(__name__)


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_atts = []

    for i in range(0, num_text, text_bs):
        batch = texts[i:min(num_text, i + text_bs)]
        text_input = surgclip.tokenize(
            batch,
            tokenizer,
            max_length=max_txt_l,
            device=device,
        )
        text_feat = model.encode_text(text_input)[0]
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    return torch.cat(text_feats, dim=0), torch.cat(text_atts, dim=0)


def format_results(results, preds, labels, video_idxs, frame_idxs):
    # One unique label per frame
    if len(labels.shape) == 1:
        labels = labels.tolist()
    # More than one label per frame
    elif len(labels.shape) == 2:
        result = []
        for row in labels:
            # Filter out -1 elements
            filtered_row = [x.item() for x in row if x != -1]
            result.append(filtered_row)
        labels = result

    for pred, label, video_idx, frame_idx in zip(preds, labels, video_idxs, frame_idxs):
        results.append({
            'video_idx': video_idx.item(),
            'frame_idx': frame_idx.item(),
            'prediction': pred.cpu().tolist(),
            'ground_truth': label
        })
    return results


def inference(data_loader, model, device, config, text_feats, text_atts):

    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting image feats and computing similarity scores"
    iterator = metric_logger.log_every(data_loader, 1, header)
    video_idxs = []
    frame_idxs = []
    labels = []
    results = []
    for image, label, video_idx, frame_idx in iterator:
        video_idxs.extend(video_idx.tolist())
        frame_idxs.extend(frame_idx.tolist())
        image = image.to(device, non_blocking=True)
        image_feat, pooled_image_feat = model.encode_vision(image)
        
        if config.evaluation.eval_frame_ensemble == "concat":  # default
            image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]

        pooled_image_feat = (
            pooled_image_feat.to(device, non_blocking=True)
            if config.evaluation.eval_offload
            else pooled_image_feat
        )

        i2t_scores, _ = model.get_sim(
            model.vision_proj(pooled_image_feat), model.text_proj(text_feats[:, 0])
        )

        preds = i2t_scores
        results = format_results(results, preds, label, video_idx, frame_idx)
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(config.output_dir, f'predictions_{data_loader.dataset.name}.csv'), index=False)

    return results_df
    


def inference_triplet(data_loader, model, device, config, task_text_feats, text_atts):
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting image feats and computing similarity scores"
    iterator = metric_logger.log_every(data_loader, 1, header)
    video_idxs = []
    frame_idxs = []
    labels = []
    results = {}
    results_df = {}
    for image, label, video_idx, frame_idx in iterator:
        video_idxs.extend(video_idx.tolist())
        frame_idxs.extend(frame_idx.tolist())
        image = image.to(device, non_blocking=True)
        image_feat, pooled_image_feat = model.encode_vision(image)
        
        if config.evaluation.eval_frame_ensemble == "concat":  # default
            image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]

        pooled_image_feat = (
            pooled_image_feat.to(device, non_blocking=True)
            if config.evaluation.eval_offload
            else pooled_image_feat
        )

        
        for task, text_feats in task_text_feats.items():
            i2t_scores, _ = model.get_sim(
                model.vision_proj(pooled_image_feat), model.text_proj(text_feats[0][:, 0])
            )
            preds = i2t_scores
            label_task = torch.stack(label[task]).T

            if task not in results:
                results[task] = []
                results_df[task] = []

            results[task] = format_results(results[task], preds, label_task, video_idx, frame_idx)
            results_df[task] = pd.DataFrame(results[task])
            results_df[task].to_csv(os.path.join(config.output_dir, f'predictions_{task}_{data_loader.dataset.name}.csv'), index=False)


    return results_df



@torch.no_grad()
def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix="", sota=False, probing=False, temporal_probing=False):
    categories = data_loader.dataset.categories
    task = data_loader.dataset.task

    if data_loader.dataset.name == 'cholect50':
        results_df = evaluation_triplet(
            model, data_loader, tokenizer, device, config
        )

    else:
        with torch.cuda.amp.autocast(enabled=config.fp16):
            results_df = evaluation(
                    model, data_loader, tokenizer, device, config
                )

    if task in ['phases', 'steps']:
        evaluator = WorkflowEvaluator(phases=categories, prefix=prefix)
    elif task in ['instruments']:
        evaluator = ToolPresenceEvaluator(tools=categories, prefix=prefix)
    elif task in ['triplet']:
        evaluator = TripletEvaluator(
            tasks=list(data_loader.dataset.prompts.keys()), 
            prefix=prefix
            )

    res = evaluator.evaluate(results_df)
    return res



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.fp16 else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation for media_type={media_type}")

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.prompts
    logger.info("PROMPTS: ----------")
    logger.info(texts)
    max_txt_l = config.inputs.max_txt_l
    if not isinstance(max_txt_l, int):
        max_txt_l = max_txt_l[media_type]
    
    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)
    
    results_df = inference(
        data_loader, model, device, config, text_feats, text_atts
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    
    logger.info("Finished Inference")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")
    return results_df


def evaluation_triplet(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.fp16 else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation for media_type={media_type}")

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    tasks_texts = data_loader.dataset.prompts
    logger.info("PROMPTS: ----------")
    logger.info(tasks_texts)
    max_txt_l = config.inputs.max_txt_l
    if not isinstance(max_txt_l, int):
        max_txt_l = max_txt_l[media_type]
    
    tasks_text_feats = {}
    for task_name, texts in tasks_texts.items():
        text_feats, text_atts = extract_text_feats(
            texts, max_txt_l, tokenizer, model, device
        )  # (bsz, Lt, d), (bsz, Lt)
        tasks_text_feats[task_name] = (text_feats, text_atts)
    
    results_df = inference_triplet(
        data_loader, model, device, config, tasks_text_feats, text_atts
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    
    logger.info("Finished Inference")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")
    return results_df