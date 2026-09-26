#!/bin/bash
#SBATCH --job-name="Smoothed_Diff"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --export=ALL
#SBATCH --time=09:00:00
#SBATCH --output ./logs/output.%j
#SBATCH --error ./logs/error.%j
#SBATCH -p siegfried
#SBATCH -A 2001281024
#SBATCH --mail-user=zachary_katz@mines.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

cd $SLURM_SUBMIT_DIR
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Remove prior logs
#rm logs/output*
#rm logs/error*

srun uv run python Fig4Wendian.py
