import numpy as np


def calculate_avg_iou(ground_truth, binary_scores):
    ious = []

    for binary_score, gt_score in zip(binary_scores, ground_truth):
        if gt_score.any() > 0:  # when the gt have no anomaly pixels, skip it
            iou = calculate_iou(binary_score, gt_score)
            ious.append(iou)

    return np.array(ious).mean()


def calculate_iou(prediction, mask):
    intersection = np.logical_and(prediction, mask).astype(np.float32).sum()
    union = np.logical_or(prediction, mask).astype(np.float32).sum()
    iou = intersection / union

    return iou
