#!/bin/bash
#SBATCH --job-name=bsds_hmm_compare
#SBATCH --output=logs/comparison_%j.out
#SBATCH --error=logs/comparison_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
# Note: These are CPU-only jobs, no GPU required

# ============================================================
# BSDS vs HMM Comparison - SLURM Job Script
# ============================================================
# Usage:
#   sbatch scripts/run_comparison.slurm [options]
#
# Submit from BSDS_Project directory:
#   cd /path/to/UNSEAM/BSDS_Project
#   sbatch scripts/run_comparison.slurm
# ============================================================

echo "============================================================"
echo "BSDS vs HMM Comparison - Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "============================================================"

# Load conda environment
source ~/.bashrc
conda activate emofilm-hmm

# Navigate to project directory
cd ${SLURM_SUBMIT_DIR}

# Create logs directory if not exists
mkdir -p logs results/comparison

# ============================================================
# Mode 1: Run on simulated data (for testing)
# ============================================================
echo ""
echo "Running comparison on simulated data..."
python compare_methods.py \
    --mode simulated \
    --n-subjects 10 \
    --n-timepoints 200 \
    --n-voxels 100 \
    --n-events 8 \
    --output results/comparison

# ============================================================
# Mode 2: Run on Emo-Film data (if available)
# ============================================================
EMOFILM_DATA="/storage/bigdata/Emo-FiLM"

if [ -d "${EMOFILM_DATA}" ]; then
    echo ""
    echo "Emo-FiLM data found at ${EMOFILM_DATA}"
    echo "Running comparison on Emo-FiLM data..."

    # First extract ROI timeseries if not already done
    python run_emofilm_bsds.py \
        --mode extract \
        --data-dir ${EMOFILM_DATA} \
        --task BigBuckBunny \
        --output results/emofilm_extracted

    # Run BSDS on Emo-Film
    python run_emofilm_bsds.py \
        --mode fit \
        --data-dir results/emofilm_extracted \
        --task BigBuckBunny \
        --n-states 6 \
        --output results/bsds_emofilm

    # Run HMM baselines on Emo-Film
    python run_hmm_baseline.py \
        --dataset emofilm \
        --data-dir results/emofilm_extracted \
        --task BigBuckBunny \
        --n-events 6 \
        --output-dir results/hmm_baseline

else
    echo ""
    echo "Emo-FiLM data not found at ${EMOFILM_DATA}"
    echo "Skipping real data analysis."
fi

echo ""
echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
