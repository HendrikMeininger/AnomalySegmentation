"""
    Script was adjusted from the official mvtec evaluation code.
    The original code is available at https://www.mvtec.com/company/research/datasets/mvtec-ad.
    Scientific Papers:
        https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
"""

import numpy as np


def compute_classification_roc(
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
    """Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain
          a real-valued anomaly score at each pixel.
        scoring_function: Function that turns anomaly maps into a single
          real valued anomaly score.

        ground_truth_labels: List of integers that indicate the ground truth
          class for each input image. 0 corresponds to an anomaly-free sample
          while a value != 0 indicates an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)
    num_scores = len(anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Initially, every NOK sample is correctly classified as anomalous
    # (tpr = 1.0), and every OK sample is incorrectly classified as anomalous
    # (fpr = 1.0).
    fprs = [1.0]
    tprs = [1.0]

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    next_score = None
    highest_accuracy = 0
    optimal_threshold = -1

    for i, (current_score, label) in enumerate(sorted_samples):
        labels = [i[1] for i in sorted_samples]

        correct = 0
        lower_labels = labels[:i]
        upper_labels = labels[i:]

        correct += lower_labels.count(0)
        correct += upper_labels.count(1)

        accuracy = correct / len(labels)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            last_scores = sorted_samples[i - 1][0] if i > 0 else current_score
            optimal_threshold = (last_scores + current_score) / 2

        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i < num_scores - 1:
            next_score = sorted_samples[i + 1][0]
        else:
            next_score = None  # end of list

        if (next_score != current_score) or (next_score is None):
            fprs.append(num_fp / num_ok)
            tprs.append(num_tp / num_nok)

    # Return (FPR, TPR) pairs in increasing order.
    fprs = fprs[::-1]
    tprs = tprs[::-1]

    # print("Highest accuracy: " + str(highest_accuracy))
    # print("Optimal threshold: " + str(optimal_threshold))

    return fprs, tprs, optimal_threshold, highest_accuracy
