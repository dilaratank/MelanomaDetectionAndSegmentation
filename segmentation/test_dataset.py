import os
import shutil
import nibabel as nib
import numpy as np

# Define the paths to the original data and target directories
data_folder = '../data/segmentation/test'
nnunet_results_folder = '/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/Dataset011_Melanoma'

# Create the test_set and test_msk folders if they don't exist
test_set_folder = os.path.join(nnunet_results_folder, 'nnUNetTrainer__nnUNetPlans__2d', 'test_set')
test_msk_folder = os.path.join(nnunet_results_folder, 'nnUNetTrainer__nnUNetPlans__2d', 'test_msk')

os.makedirs(test_set_folder, exist_ok=True)
os.makedirs(test_msk_folder, exist_ok=True)

# Function to copy and rename test data
def copy_and_rename_test_data(source_folder, target_set_folder, target_msk_folder):
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
                new_image_name = f'Dataset011_Melanoma_{image_number}_0000.{file_ending}'

                # Copy the image to the test_set folder
                shutil.copy(image_path, os.path.join(target_set_folder, new_image_name))

                # Optional: If you want to manipulate the image, you can add code here

                # Example code to manipulate the image using Nibabel
                img = nib.load(image_path).get_fdata()
                img = img[:, :, 0]

                img = nib.Nifti1Image(img, np.eye(4))
                img.get_data_dtype() == np.dtype(np.int16)
                img.header.get_xyzt_units()

                nib.save(img, os.path.join(target_set_folder, new_image_name))

            elif file.startswith('msk_'):
                # Mask file
                mask_path = os.path.join(root, file)

                # Get the image number from the file name
                image_number = file.split('_')[1].split('.')[0]

                # Determine the file ending based on the image format (e.g., .nii.gz)
                file_ending = "nii.gz"

                # Construct the new mask file name
                new_mask_name = f'Dataset011_Melanoma_{image_number}.{file_ending}'

                # Copy the mask to the test_msk folder
                shutil.copy(mask_path, os.path.join(target_msk_folder, new_mask_name))

                # Optional: If you want to manipulate the mask, you can add code here

                # Example code to manipulate the mask using Nibabel
                # mask = nib.load(mask_path).get_fdata()
                # mask = np.expand_dims(mask, axis=2)

                # img = nib.Nifti1Image(mask, np.eye(4))
                # img.get_data_dtype() == np.dtype(np.int16)
                # img.header.get_xyzt_units()

                # nib.save(img, os.path.join(target_msk_folder, new_mask_name))

# Copy and rename test data from the source folder to the target folders
copy_and_rename_test_data(data_folder, test_set_folder, test_msk_folder)

print("Test data preparation completed.")
