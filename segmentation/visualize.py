import cv2
import os
import matplotlib as plt
import numpy as np
from datasets import Scan_Dataset_Segm

data_dir = '/home/scur0404/projects/MelanomaDetectionAndSegmentation/data_processed/segmentation'

#@title Visualizing Images { run: 'auto'}
nn_set = 'train' #@param ['train', 'val', 'test']
index  = 0 #@param {type:'slider', min:0, max:120, step:5}

# import cv2
dataset  = Scan_Dataset_Segm(os.path.join(data_dir, nn_set))

sample = dataset[1]
print(sample['mask'].shape)


# print(len(dataset))
# print(os.path.join(data_dir, nn_set))

# n_images_display = 5
# fig, ax = plt.subplots(1, n_images_display, figsize=(20, 5))
# for i in range(n_images_display):
#   sample = dataset[index+i]
#   image, mask = sample['image'], sample['mask']
#   image, mask = np.uint8(image), np.uint8(mask)
#   edge = cv2.Canny(mask*255, 100, 200)
#   image[edge>0] = (0, 255, 0)
#   ax[i].imshow(image)
#   ax[i].axis('off')
# plt.suptitle(f"Skin Cancer Images [Indices: {index} - {index + n_images_display} - Images Shape: {dataset[0]['image'].shape}]");