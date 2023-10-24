#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=nnunettrain
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

# Execute program located in $HOME and redirect outputs to the log file
nnUNetv2_train 13 2d all


# nnUNetv2_train 11 2d 1
# srun python train_classifier.py --batch_size=64 --lr=0.02 --freeze_until_layer=5


# nnUNetv2_predict -i /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_raw/Dataset012_Melanoma -o /home/scur0404/projects/MelanomaDetectionAndSegmentation/nnUNet_results/try_results -d 12 -c 2d

# nnUNetTrainer_5epochs