import numpy as np
from bisect import bisect

"""
    Code was adjusted from the official mvtec evaluation code.
    The original code is available at https://www.mvtec.com/company/research/datasets/mvtec-ad.
    Scientific Papers:
        https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
"""


def calculate_auc(x_values, y_values, x_max=None):
    """
    This function calculates the definite integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x_values: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y_values: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside the range of x.

    Returns:
        Area under the curve.
    """

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    finite_mask = np.logical_and(np.isfinite(x_values), np.isfinite(y_values))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x_values:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x_values, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x_values)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y_values[ins - 1] + ((y_values[ins] - y_values[ins - 1]) *
                                            (x_max - x_values[ins - 1]) /
                                            (x_values[ins] - x_values[ins - 1]))
            correction = 0.5 * (y_interp + y_values[ins - 1]) * (x_max - x_values[ins - 1])

        # Cut off at x_max.
        mask = x_values <= x_max
        x_values = x_values[mask]
        y_values = y_values[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y_values[1:] + y_values[:-1]) * (x_values[1:] - x_values[:-1])) + correction
