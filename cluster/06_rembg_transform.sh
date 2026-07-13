#!/bin/bash
#SBATCH --job-name=06_run_rembg_transform
#SBATCH --output=../logs/06_run_rembg_transform_%j.out
#SBATCH --error=../logs/06_run_rembg_transform_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --wckey=noop


# Main job command
source $(conda info --base)/etc/profile.d/conda.sh

conda activate master_thesis

python \
    /storage/homefs/kw23y068/Master_Thesis/scripts/rembg_transform_v01.py \
    /storage/homefs/kw23y068/Master_Thesis/config.yaml