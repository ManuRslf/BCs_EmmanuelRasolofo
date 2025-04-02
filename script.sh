#!/bin/bash
#SBATCH --job-name=TrainingDinov2-llma  
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4               
#SBATCH --gres gpu:1,VramPerGpu:70G                    
#SBATCH --time=1-00:00:00                 
#SBATCH --partition=public-gpu               
#SBATCH --mem=32000                      


module load CUDA/12.1                  
module load cuDNN



srun python3 main.py