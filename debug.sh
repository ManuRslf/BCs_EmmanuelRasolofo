#!/bin/bash
#SBATCH --job-name=TrainingDinov2-llma  
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4               
#SBATCH --gpus=1                
#SBATCH --time=00:15:00                 
#SBATCH --partition=debug-gpu               
#SBATCH --mem=8000                      


module load CUDA/12.1                  
module load cuDNN



srun python3 utils.py