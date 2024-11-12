import cv2
import numpy as np

"""def crop_image(image):
    for w_pos in reversed(range(image.shape[1])):
        if (image[:, w_pos] == [255, 255, 255]).all():
            image = np.delete(image, w_pos, 1)
    for h_pos in reversed(range(image.shape[0])):
        if (image[h_pos, :] == [255, 255, 255]).all():
            image = np.delete(image, h_pos, 0)

    return image"""

def crop_image(image):
    a = []
    for w_pos in range(image.shape[1]):
        if (image[:, w_pos] == [255, 255, 255]).all():
            a.append(w_pos)
        elif not (image[:, w_pos] == [255, 255, 255]).all() and w_pos == 0:
            a.append(w_pos)

    if a[-1] < image.shape[1]:
        a.append(image.shape[1])

    indes = []
    
    for i in range(len(a) - 1):
        if a[i + 1] - a[i] > 500:
            indes.append(i)
    
    x_range = []
    for i in range(len(indes) -1):
        if a[indes[i+1]] - a[indes[i]+1] < 50:
            min_v = min(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
            max_v = max(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
            x_range.append([min_v, max_v])

    if x_range == []:
        for i in range(len(indes)):
            x_range.append([a[indes[i]], a[indes[i] + 1]])
    
    width_range = x_range[0]
    croping_img = image[:, :width_range[1], :]

    for h_pos in reversed(range(croping_img.shape[0])):
        if (croping_img[h_pos, :] == [255, 255, 255]).all():
            croping_img = np.delete(croping_img, h_pos, 0)

    return croping_img