#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

module purge
module load Python/3.10.8-GCCcore-12.2.0

VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment setup complete."
