import cv2
import numpy as np
import scipy.ndimage as ndi

def remove_background(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    th = cv2.threshold(
        a, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    mask = np.zeros_like(a)
    mask[a < th] = 1
    mask[a >= th] = 2
    mask = ndi.binary_fill_holes(mask-1)

    masked_image = np.zeros_like(image)
    masked_image[mask == 1] = image[np.where(mask == 1)]
    masked_image[mask == 0] = 255.

    return masked_image

def background_delete(image):
    for w_pos in reversed(range(image.shape[1])):
        if (image[:, w_pos] == [255, 255, 255]).all():
            image = np.delete(image, w_pos, 1)
    
    for h_pos in reversed(range(image.shape[0])):
        if (image[h_pos, :] == [255, 255, 255]).all():
            image = np.delete(image, h_pos, 0)

    return image
