#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=32000M

source $HOME/melanomadetseg/bin/activate

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

export nnUNet_raw="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_raw"
export nnUNet_preprocessed="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_preprocessed"
export nnUNet_results="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results"

cd /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet

# Execute program
# train
#nnUNetv2_train 11 2d all

# inference
nnUNetv2_predict -i /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/Dataset011_Melanoma/nnUNetTrainer__nnUNetPlans__2d/test_set -o /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/Dataset011_Melanoma/nnUNetTrainer__nnUNetPlans__2d/inference_preds -d 11 -c 2d -f all