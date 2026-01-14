#!/bin/bash
#SBATCH --job-name=hmm_emofilm
#SBATCH --output=logs/hmm_emofilm_%j.out
#SBATCH --error=logs/hmm_emofilm_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal

# ============================================================
# HMM Event Segmentation on Emo-Film Data
# ============================================================
# Usage:
#   sbatch scripts/run_hmm_emofilm.slurm
#   sbatch scripts/run_hmm_emofilm.slurm --export=TASK=FirstBite
# ============================================================

echo "============================================================"
echo "HMM Emo-Film Analysis - Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================================"

# Load environment
source ~/.bashrc
conda activate emofilm-hmm

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs results/hmm_baseline

# Default task
TASK=${TASK:-BigBuckBunny}
N_EVENTS=${N_EVENTS:-8}
EMOFILM_DATA="/storage/bigdata/Emo-FiLM"

echo "Task: ${TASK}"
echo "N_EVENTS: ${N_EVENTS}"
echo "Data path: ${EMOFILM_DATA}"

# Run HMM baseline
python run_hmm_baseline.py \
    --dataset emofilm \
    --data-dir ${EMOFILM_DATA} \
    --task ${TASK} \
    --n-events ${N_EVENTS} \
    --n-iter 100 \
    --output-dir results/hmm_baseline \
    --verbose

echo ""
echo "Job completed at $(date)"
