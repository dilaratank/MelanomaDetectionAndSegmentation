# -*- coding: utf-8 -*-
from google.colab.patches import cv2_imshow
import cv2
import shutil

"""
Following are the DHR tasks followed in this example code:

    -- Applying Morphological Black-Hat transformation
    -- Creating the mask for InPainting task
    -- Applying inpainting algorithm on the image

    Taken from https://github.com/sunnyshah2894/DigitalHairRemoval/blob/master/DigitalHairRemoval.py
"""
dataset  = Scan_Dataset_Segm('/content/gdrive/MyDrive/data/segmentation/train')

n_images_display = 1
print(len(dataset))
for i in range(len(dataset)):
  if i % 100 == 0:
    print(i)
  sample = dataset[i]
  image, image_name, mask, mask_name = sample['image'][0], sample['image'][1], sample['mask'][0], sample['mask'][1]
  image = cv2.convertScaleAbs(image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  mask_path = mask_name
  mask_name = mask_name.split('/')[-1].split('.')[0]

  location_to = '/content/gdrive/MyDrive/data vandaag processed/segmentation/train/'
  mask_to_path = f'{location_to}/{mask_name}.nii.gz'

  shutil.copy(mask_path, mask_to_path)

  # img_nifti = nib.Nifti1Image(mask, np.eye(4))  # Assuming the affine matrix is identity
  # nib.save(img_nifti, f'/content/gdrive/MyDrive/data_processed/segmentation/test/{mask_name}.nii.gz')
  #print('saved mask')

  image_name = image_name.split('/')[-1].split('.')[0]
  # Convert the original image to grayscale
  grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # Kernel for the morphological filtering
  kernel = cv2.getStructuringElement(1, (17, 17))

  # Perform the blackHat filtering on the grayscale image to find the hair countours
  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

  # intensify the hair countours in preparation for the inpainting algorithm
  ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

  # inpaint the original image depending on the mask
  dst = cv2.inpaint(image, thresh2, 1, cv2.INPAINT_TELEA)

  # cv2_imshow(dst)

  # Save the processed image as .nii.gz
  img_nifti = nib.Nifti1Image(dst, np.eye(4))  # Assuming the affine matrix is identity
  nib.save(img_nifti, f'/content/gdrive/MyDrive/data vandaag processed/segmentation/train/{image_name}.nii.gz')
  # #print('saved image')