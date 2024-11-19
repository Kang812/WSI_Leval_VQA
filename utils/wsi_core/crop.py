import cv2
import numpy as np

def crop_image(img, img_mask, mask):
    a = []
    for w_pos in range(img_mask.shape[1]):
        if (img_mask[:, w_pos] == [255, 255, 255]).all():
            a.append(w_pos)
        elif not (img_mask[:, w_pos] == [255, 255, 255]).all() and w_pos == 0:
            a.append(w_pos)

    if a[-1] < img_mask.shape[1]:
        a.append(img_mask.shape[1])

    indes = []
    for i in range(len(a) - 1):
        if a[i + 1] - a[i] > 500:
            indes.append(i)
    
    if indes == []:
        for i in range(len(a) - 1):
            if a[i + 1] - a[i] > 300:
                indes.append(i)

    if indes == []:
        for i in range(len(a) - 1):
            if a[i + 1] - a[i] > 100:
                indes.append(i)
    x_range = []
    if len(indes) <= 3:
        for i in range(len(indes)):
            x_range.append([a[indes[i]], a[indes[i] + 1]])

    if x_range == []:
        for i in range(len(indes) -1):
            if a[indes[i+1]] - a[indes[i]+1] < 50:
                min_v = min(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
                max_v = max(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
                x_range.append([min_v, max_v])

    if x_range == []:
        for i in range(len(indes) -1):
            if a[indes[i+1]] - a[indes[i]+1] < 300:
                min_v = min(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
                max_v = max(a[indes[i+1]], a[indes[i+1] + 1], a[indes[i]+1], a[indes[i]])
                x_range.append([min_v, max_v])

    if x_range == []:
        for i in range(len(indes)):
            x_range.append([a[indes[i]], a[indes[i] + 1]])
    
    #width_range = x_range[0]
    
    label_counts = 0
    for i, crop_p in enumerate(x_range):
        crop_mask = mask[:, crop_p[0] + 50:crop_p[1] + 50]
        label, label_cnt = np.unique(crop_mask, return_counts = True)
        if 1 in list(label):
            if label_cnt[1] > label_counts:
                label_cnt = label_cnt[1]
                max_index = i
    
    width_range = x_range[max_index]
    croping_img = img[:, width_range[0] + 50 : width_range[1] + 50, :]
    croping_img_mask = img_mask[:, width_range[0] + 50 : width_range[1] + 50, :]
    crop_mask = mask[:, width_range[0] + 50 : width_range[1] + 50]

    for w_pos in reversed(range(croping_img.shape[1])):
        if (croping_img_mask[:, w_pos] == [255, 255, 255]).all():
            croping_img = np.delete(croping_img, w_pos, 1)
            crop_mask = np.delete(crop_mask,  w_pos, 1)
    for h_pos in reversed(range(croping_img.shape[0])):
        if (croping_img_mask[h_pos, :] == [255, 255, 255]).all():
            croping_img = np.delete(croping_img, h_pos, 0)
            crop_mask = np.delete(crop_mask,  h_pos, 0)

    return croping_img, crop_mask

if __name__ == '__main__':
    import cv2
    import os
    from glob import glob
    from mask_gen import remove_background

    mask_paths = glob("/workspace/whole_slide_image_LLM/data/train_masks/*.png")
    for mask_path in mask_paths:
        img_path = mask_path.replace("train_masks", "train_imgs")
    
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        mask[mask != 255] = 3
        mask[mask == 255] = 0
        mask[mask == 3] = 1
        
        img_mask = remove_background(img)
        croping_img, crop_mask = crop_image(img, img_mask, mask)

        cv2.imwrite(os.path.join("/workspace/whole_slide_image_LLM/data/classification_dataset/crop_image/", img_path.split("/")[-1]), croping_img)
        cv2.imwrite(os.path.join("/workspace/whole_slide_image_LLM/data/classification_dataset/crop_mask/", img_path.split("/")[-1]), crop_mask)