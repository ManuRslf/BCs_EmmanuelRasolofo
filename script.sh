#!/bin/bash
#SBATCH --job-name=TrainingDinov2-llma  
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4               
#SBATCH --gpus=2                    
#SBATCH --time=10:00:00                 
#SBATCH --partition=shared-gpu               
#SBATCH --mem=16000                      


module load CUDA/12.1                  
module load cuDNN



srun python3 models.py