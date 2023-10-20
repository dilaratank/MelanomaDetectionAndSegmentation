#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=nnUnet
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=32000M

source $HOME/melanomadetseg/bin/activate

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

export nnUNet_raw="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_raw"
export nnUNet_preprocessed="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_preprocessed"
export nnUNet_results="/home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results"

cd /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet

# Execute program located in $HOME and redirect outputs to the log file
nnUNetv2_train 11 2d 0
# srun python train_classifier.py --batch_size=64 --lr=0.02 --freeze_until_layer=5
