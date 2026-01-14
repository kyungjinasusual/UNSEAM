#!/bin/bash
#SBATCH --job-name=validate_baldassano
#SBATCH --output=logs/validate_baldassano_%j.out
#SBATCH --error=logs/validate_baldassano_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal

# ============================================================
# Baldassano 2017 Replication - HMM Validation
# ============================================================
# Usage:
#   sbatch scripts/validate_baldassano.slurm
#
# Environment variables:
#   SHERLOCK_DATA: Path to Sherlock dataset (default: ~/data/sherlock)
#   ROI: Region of interest (default: AG)
#   N_EVENTS: Number of events (default: 25)
# ============================================================

echo "============================================================"
echo "Baldassano 2017 Replication - Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "============================================================"

# Load environment
source ~/.bashrc
conda activate unseam-validation 2>/dev/null || conda activate emofilm-hmm

cd ${SLURM_SUBMIT_DIR}/..
mkdir -p logs results/validation/baldassano

# Default parameters
SHERLOCK_DATA=${SHERLOCK_DATA:-~/data/sherlock}
ROI=${ROI:-AG}
N_EVENTS=${N_EVENTS:-25}
N_ITER=${N_ITER:-100}
TOLERANCE=${TOLERANCE:-3}

echo ""
echo "Parameters:"
echo "  Data path: ${SHERLOCK_DATA}"
echo "  ROI: ${ROI}"
echo "  N_EVENTS: ${N_EVENTS}"
echo "  N_ITER: ${N_ITER}"
echo "  Tolerance: ${TOLERANCE} TRs"
echo ""

# Check data exists
if [ ! -d "${SHERLOCK_DATA}" ]; then
    echo "ERROR: Sherlock data directory not found at ${SHERLOCK_DATA}"
    echo ""
    echo "To download Sherlock data:"
    echo "  mkdir -p ~/data/sherlock && cd ~/data/sherlock"
    echo "  wget -O sherlock.zip 'https://figshare.com/ndownloader/files/9021937'"
    echo "  unzip sherlock.zip"
    exit 1
fi

# Run validation
python scripts/validate_baldassano.py \
    --data-dir ${SHERLOCK_DATA} \
    --roi ${ROI} \
    --n-events ${N_EVENTS} \
    --n-iter ${N_ITER} \
    --tolerance ${TOLERANCE} \
    --method both \
    --output-dir results/validation/baldassano

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Validation completed successfully at $(date)"
    echo "Results saved to: results/validation/baldassano/"
else
    echo "Validation FAILED with exit code: ${exit_code}"
fi
echo "============================================================"

exit $exit_code
