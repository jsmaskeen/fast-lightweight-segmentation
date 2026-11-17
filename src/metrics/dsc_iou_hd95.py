import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from typing import Dict, Tuple


def compute_dsc_iou_hd95(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    image_shape: Tuple[int, int] = (256, 256)
) -> Dict[str, float]:
    """
    Compute DSC, IoU, and HD95 between binary prediction and ground-truth masks.
    - Masks must be 2D boolean arrays or uint8 {0, 1} / {0, 255}.
    - HD95 convention:
      - If both empty: excluded (NaN)
      - If prediction is empty but GT is not: set to diagonal (256 * sqrt(2) ~ 362.0387 px)
    """
    p = (pred_mask > 0).astype(bool)
    y = (target_mask > 0).astype(bool)

    p_sum = int(p.sum())
    y_sum = int(y.sum())
    intersection = int((p & y).sum())
    union = int((p | y).sum())

    # DSC & IoU calculation
    if p_sum == 0 and y_sum == 0:
        dsc = 1.0
        iou = 1.0
    elif p_sum == 0 or y_sum == 0:
        dsc = 0.0
        iou = 0.0
    else:
        dsc = (2.0 * intersection) / (p_sum + y_sum)
        iou = intersection / union

    # HD95 calculation
    diag = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
    is_empty_pred = False

    if p_sum == 0 and y_sum == 0:
        hd95 = np.nan
    elif p_sum == 0 and y_sum > 0:
        hd95 = diag
        is_empty_pred = True
    elif p_sum > 0 and y_sum == 0:
        hd95 = diag
    else:
        b_p = p ^ binary_erosion(p)
        b_y = y ^ binary_erosion(y)
        if not np.any(b_p) or not np.any(b_y):
            hd95 = diag
        else:
            d_to_y = distance_transform_edt(~b_y)[b_p]
            d_to_p = distance_transform_edt(~b_p)[b_y]
            all_d = np.concatenate([d_to_y, d_to_p])
            hd95 = float(np.percentile(all_d, 95))

    return {
        "dsc": float(dsc),
        "iou": float(iou),
        "hd95": float(hd95),
        "is_empty_pred": 1.0 if is_empty_pred else 0.0,
        "pred_area_px": float(p_sum),
        "gt_area_px": float(y_sum),
    }


def compute_normal_fpr(pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute false-positive status on a normal image:
    - fpr_50px: 1.0 if predicted lesion area > 50 pixels, else 0.0
    - fpr_0px: 1.0 if predicted lesion area > 0 pixels, else 0.0
    """
    p = (pred_mask > 0).astype(bool)
    area = int(p.sum())
    return {
        "pred_area_px": float(area),
        "fp_50px": 1.0 if area > 50 else 0.0,
        "fp_0px": 1.0 if area > 0 else 0.0,
    }
