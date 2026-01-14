#!/bin/bash
#SBATCH --job-name=validate_yang
#SBATCH --output=logs/validate_yang_%j.out
#SBATCH --error=logs/validate_yang_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=normal

# ============================================================
# Yang 2023 Replication - GaussianHMM Validation
# ============================================================
# Usage:
#   sbatch scripts/validate_yang.slurm
#
# Environment variables:
#   STUDYFORREST_DATA: Path to StudyForrest dataset
#   N_STATES: Number of HMM states (default: 4)
#   RUN: Which run to analyze (default: 1)
# ============================================================

echo "============================================================"
echo "Yang 2023 Replication - Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "============================================================"

# Load environment
source ~/.bashrc
conda activate unseam-validation 2>/dev/null || conda activate emofilm-hmm

cd ${SLURM_SUBMIT_DIR}/..
mkdir -p logs results/validation/yang

# Default parameters
STUDYFORREST_DATA=${STUDYFORREST_DATA:-~/data/studyforrest}
N_STATES=${N_STATES:-4}
RUN=${RUN:-1}
N_ITER=${N_ITER:-100}
COV_TYPE=${COV_TYPE:-diag}

echo ""
echo "Parameters:"
echo "  Data path: ${STUDYFORREST_DATA}"
echo "  N_STATES: ${N_STATES}"
echo "  Run: ${RUN}"
echo "  Covariance type: ${COV_TYPE}"
echo "  N_ITER: ${N_ITER}"
echo ""

# Check data exists
if [ ! -d "${STUDYFORREST_DATA}" ]; then
    echo "ERROR: StudyForrest data directory not found at ${STUDYFORREST_DATA}"
    echo ""
    echo "To download StudyForrest data:"
    echo "  pip install gdown"
    echo "  gdown --folder https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t"
    exit 1
fi

# Run validation with model selection and cross-validation
python scripts/validate_yang.py \
    --data-dir ${STUDYFORREST_DATA} \
    --run ${RUN} \
    --n-states ${N_STATES} \
    --covariance-type ${COV_TYPE} \
    --n-iter ${N_ITER} \
    --select-n-states \
    --cross-validate \
    --output-dir results/validation/yang

exit_code=$?

echo ""
echo "============================================================"
if [ $exit_code -eq 0 ]; then
    echo "Validation completed successfully at $(date)"
    echo "Results saved to: results/validation/yang/"
else
    echo "Validation FAILED with exit code: ${exit_code}"
fi
echo "============================================================"

exit $exit_code
