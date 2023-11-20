import cv2
import numpy as np


def fuse_images(src, subst_img, subst_mask):
    if src.shape[:2] != subst_img.shape[:2] or src.shape[:2] != subst_mask.shape[:2]:
        raise ValueError("Input images and mask must have the same dimensions.")
    result = np.copy(src)

    subst_mask = subst_mask.astype(np.uint8) * 255
    original_mask = cv2.bitwise_not(subst_mask)

    result[:, :, 0] = cv2.bitwise_and(result[:, :, 0], original_mask)
    result[:, :, 1] = cv2.bitwise_and(result[:, :, 1], original_mask)
    result[:, :, 2] = cv2.bitwise_and(result[:, :, 2], original_mask)

    result[:, :, 0] += cv2.bitwise_and(subst_img[:, :, 0], subst_mask)
    result[:, :, 1] += cv2.bitwise_and(subst_img[:, :, 1], subst_mask)
    result[:, :, 2] += cv2.bitwise_and(subst_img[:, :, 2], subst_mask)

    return result
