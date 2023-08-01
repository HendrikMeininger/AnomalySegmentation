import numpy as np

from source.evaluation import util, iou
from source.evaluation import roc_metric
from source.evaluation import pro_metric
from source.evaluation import dfc_metric
from sklearn.metrics import roc_auc_score

"""
    Code is based on the official evaluation code for mvtec datasets 
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    
    Scientific Papers:
        https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
"""


def print_metrics(scores, masks, binary_scores=None):
    if binary_scores is None:
        binary_scores = calculate_binary_scores(scores)

    image_level_roc, pixel_level_roc = calculate_au_roc(ground_truth=masks,
                                                        predictions=scores)
    au_pro, pro_curve = calculate_au_pro(ground_truth=masks, predictions=scores,
                                         integration_limit=0.3)
    avg_iou = calculate_avg_iou(ground_truth=masks, binary_scores=binary_scores)

    print("Image level ROC-AUC: ", image_level_roc)
    print("Pixel level ROC-AUC: ", pixel_level_roc)
    print("PRO-AUC: ", au_pro)
    print("IoU: ", avg_iou)

    print(f"({round(pixel_level_roc * 100, 2)}, {round(au_pro * 100, 2)}, {round(avg_iou * 100, 2)})")


def calculate_binary_scores(scores):
    binary_scores = []
    for score in scores:
        bin_score = get_binary_score(score)
        binary_scores.append(bin_score)

    return binary_scores


def get_binary_score(score):
    if not np.all(score == 0):
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
    binary_score = np.zeros_like(score)
    binary_score[score <= 0.5] = 0
    binary_score[score > 0.5] = 1

    return binary_score


def calculate_dfc_metrics(ground_truth, predictions):
    dfc_metric.eval_dfc_metrics(ground_truth, predictions)


def calculate_au_roc(ground_truth, predictions):
    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]

    # official MvTec eval code does the same calculations as sklearn.metrics.roc_auc_score but is slower
    """# Compute the classification ROC curve.
    fprs, tprs, optimal_threshold, highest_accuracy = roc_metric.compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    roc_curve = fprs, tprs

    # Compute the area under the classification ROC curve.
    au_roc = util.calculate_auc(roc_curve[0], roc_curve[1])"""

    image_level_roc = roc_auc_score(binary_labels, np.asarray(predictions).max(1).max(1))
    pixel_level_roc = roc_auc_score(np.asarray(ground_truth).astype(np.bool).ravel(), np.asarray(predictions).ravel())

    return image_level_roc, pixel_level_roc


def calculate_au_pro(ground_truth, predictions, integration_limit):
    pro_curve = pro_metric.compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = util.calculate_auc(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    return au_pro, pro_curve


"""
    returns TP_rate, FP_rate, FN_rate, TP_rate, Accuracy
    
    0 = normal
    1 = abnormal
    
        |   GT  |   Prediction
    ----------------------------
    TP  |   0   |       0
    FP  |   1   |       0
    TN  |   1   |       1
    FN  |   0   |       1
"""


def calculate_rates(ground_truth, prediction_labels):
    gt_labels = [int(np.any(x > 0)) for x in ground_truth]

    n_tp = 0
    n_fp = 0
    n_fn = 0
    n_tn = 0

    for gt, pred in zip(gt_labels, prediction_labels):
        if gt == 0 and pred == 0:
            n_tp += 1
        if gt == 0 and pred == 1:
            n_fn += 1
        if gt == 1 and pred == 0:
            n_fp += 1
        if gt == 1 and pred == 1:
            n_tn += 1

    accuracy = (n_tp + n_tn) / len(prediction_labels)
    return n_tp / len(prediction_labels), n_fp / len(prediction_labels), n_fn / len(prediction_labels), n_tn / len(
        prediction_labels), accuracy


def calculate_avg_iou(ground_truth, binary_scores):
    return iou.calculate_avg_iou(ground_truth, binary_scores)
