import os

import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure
import pandas as pd


def eval_dfc_metrics(masks, scores, max_step=200, expect_fpr=0.3):
    # as array
    masks = np.array(masks)
    scores = np.array(scores)

    # binary masks
    masks[masks <= 0.5] = 0
    masks[masks > 0.5] = 1
    masks = masks.astype(np.bool)

    # auc score (image level) for detection
    labels = masks.any(axis=1).any(axis=1)
    # preds = scores.mean(1).mean(1)
    preds = scores.max(1).max(1)  # for detection max score or mean score?
    det_auc_score = roc_auc_score(labels, preds)
    det_pr_score = average_precision_score(labels, preds)

    # auc score (per pixel level) for segmentation
    seg_auc_score = roc_auc_score(masks.ravel(), scores.ravel())
    seg_pr_score = average_precision_score(masks.ravel(), scores.ravel())
    # metrics over all data
    print(f"Det AUC: {det_auc_score:.4f}, Seg AUC: {seg_auc_score:.4f}")
    print(f"Det PR: {det_pr_score:.4f}, Seg PR: {seg_pr_score:.4f}")

    # per region overlap and per image iou
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        if step % 100 == 0:
            print('Step: ', step)

        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]    # bug
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            if masks[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        # print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)

    # save results
    data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
    df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                    'pros_mean', 'pros_std',
                                                    'ious_mean', 'ious_std'])

    # best per image iou
    best_miou = ious_mean.max()
    print(f"Best IOU: {best_miou:.4f}")

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]

    pros_mean_selected = pros_mean[idx]  # no scale  (correct?)

    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)

    return f"{det_pr_score:.5f},{det_auc_score:.5f},{seg_pr_score:.5f},{seg_auc_score:.5f},{pro_auc_score:.5f},{best_miou:.5f}"


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
