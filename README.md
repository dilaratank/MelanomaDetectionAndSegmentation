# MelanomaDetectionAndSegmentation
Code to reporduce our Melanoma Detection and Segmentation for MAM11A. 

## Detection (Classification)
The script in the classification folder can be run directly on a cluster with the bash scripts. 
Test output scores are written to .out files that you can find in the folder as well. 

## Segmentation
The script in the segmentation folder can be run directly on a cluster with the bash scripts to run a training on a simple unet.

For the nnUNet, we give the scripts that we use to structure our dataset. 
Further commands to use the nnUNet are

nnUNetv2_plan_and_preprocess -d 11_ID --verify_dataset_integrity

and

python evaluate_predictions.py /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/Dataset011_Melanoma/nnUNetTrainer__nnUNetPlans__2d/test_msk /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/Dataset011_Melanoma/nnUNetTrainer__nnUNetPlans__2d/inference_preds -l 0 1


The unet and nnUNet output scores are written to output files in the segmentation folder in an .out and .json format.