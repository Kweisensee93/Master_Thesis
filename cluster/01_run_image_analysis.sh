#!/bin/bash
#SBATCH --job-name=01_run_image_analysis
#SBATCH --output=../logs/01_run_image_analysis_%j.out
#SBATCH --error=../logs/01_run_image_analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --wckey=noop

conda activate master_thesis

python ../scripts/remg_v05.py