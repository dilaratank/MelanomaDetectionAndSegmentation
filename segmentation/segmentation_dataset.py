import os
import json
import shutil
import nibabel as nib
import numpy as np

# Define the paths to your original data and the target nnUNet_raw folder
data_folder = '../data'
nnunet_raw_folder = '../nnUNet_raw'

# Create the nnUNet_raw folder if it doesn't exist
if not os.path.exists(nnunet_raw_folder):
    os.makedirs(nnunet_raw_folder)

# Function to copy images and masks to the target directory with the proper naming convention
def copy_images_and_masks(source_folder, target_folder, dataset_id):
    # Create subfolders for imagesTr and labelsTr
    images_tr_folder = os.path.join(target_folder, f'Dataset{dataset_id}_Melanoma', 'imagesTr')
    labels_tr_folder = os.path.join(target_folder, f'Dataset{dataset_id}_Melanoma', 'labelsTr')

    os.makedirs(images_tr_folder, exist_ok=True)
    os.makedirs(labels_tr_folder, exist_ok=True)

    # Iterate through the source data
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.startswith('img_'):
                # Image file
                image_path = os.path.join(root, file)

                # Get the image number from the file name
                image_number = file.split('_')[1].split('.')[0]

                # Determine the file ending based on the image format (e.g., .nii.gz)
                file_ending = "nii.gz"

                # Construct the new image file name
                new_image_name = f'Dataset{dataset_id}_Melanoma_{image_number}_0000.{file_ending}'

                # Copy the image to the imagesTr folder
                shutil.copy(image_path, os.path.join(images_tr_folder, new_image_name))

                img = nib.load(image_path).get_fdata()
                img = img[:, :, 0]

                img = nib.Nifti1Image(img, np.eye(4))
                img.get_data_dtype() == np.dtype(np.int16)
                img.header.get_xyzt_units()

                nib.save(img, os.path.join(images_tr_folder, new_image_name))


            elif file.startswith('msk_'):
                # Mask file
                mask_path = os.path.join(root, file)

                # Get the image number from the file name
                image_number = file.split('_')[1].split('.')[0]

                # Determine the file ending based on the image format (e.g., .nii.gz)
                file_ending = "nii.gz"

                # Construct the new mask file name
                new_mask_name = f'Dataset{dataset_id}_Melanoma_{image_number}.{file_ending}'

                # Copy the mask to the labelsTr folder
                shutil.copy(mask_path, os.path.join(labels_tr_folder, new_mask_name))

                # mask = nib.load(mask_path).get_fdata()
                # mask = np.expand_dims(mask, axis=2)
                # #mask = np.resize(mask, (256, 256, 3))  # hardcoded

                # img = nib.Nifti1Image(mask, np.eye(4))
                # img.get_data_dtype() == np.dtype(np.int16)
                # img.header.get_xyzt_units()

                # nib.save(img, os.path.join(labels_tr_folder, new_mask_name))

# Function to copy images and masks from the "val" folder
def copy_val_data(data_folder, nnunet_raw_folder, dataset_id):
    val_folder = os.path.join(data_folder, 'segmentation', 'val')
    copy_images_and_masks(val_folder, nnunet_raw_folder, dataset_id)

# Specify your dataset ID (choose an unused ID)
dataset_id = 11   # You can change this to an unused ID

# Copy images and masks from the "train" folder
copy_images_and_masks(os.path.join(data_folder, 'segmentation', 'train'), nnunet_raw_folder, dataset_id)

# Copy images and masks from the "val" folder
copy_val_data(data_folder, nnunet_raw_folder, dataset_id)

# Create the dataset.json file
dataset_info = {
    "channel_names": {
        "RGB": 0,
    },
    "labels": {
        "background": 0,
        "melanoma": 1,
    },
    "numTraining": len(os.listdir(os.path.join(nnunet_raw_folder, f'Dataset{dataset_id}_Melanoma', 'imagesTr'))),
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "NibabelIO"
    }

with open(os.path.join(nnunet_raw_folder, f'Dataset{dataset_id}_Melanoma', 'dataset.json'), 'w') as json_file:
    json.dump(dataset_info, json_file, indent=4)

print("Dataset preparation completed.")

