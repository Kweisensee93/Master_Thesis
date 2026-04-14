#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=../logs/setup_env_%j.out
#SBATCH --error=../logs/setup_env_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --wckey=noop

# --- Config ---
PROJECT="/storage/homefs/kw23y068/master_thesis/pipeline"
ENV_NAME="master_thesis"
PYTHON_VERSION="3.13"
REQUIREMENTS="${PROJECT}/scripts/requirements.txt"

# --- Create conda env if it doesn't exist ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}
fi

# --- Install dependencies ---
echo "Installing dependencies..."
conda run -n ${ENV_NAME} pip install -r ${REQUIREMENTS}

echo "Setup complete."