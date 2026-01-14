#!/bin/bash
#SBATCH --job-name=full_validation
#SBATCH --output=logs/full_validation_%j.out
#SBATCH --error=logs/full_validation_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal

# ============================================================
# Full HMM/BSDS Validation Pipeline
# ============================================================
# Runs both Baldassano and Yang replications sequentially
#
# Usage:
#   sbatch scripts/run_full_validation.slurm
#
# Prerequisites:
#   - Sherlock data in ~/data/sherlock or $SHERLOCK_DATA
#   - StudyForrest data in ~/data/studyforrest or $STUDYFORREST_DATA
# ============================================================

echo "============================================================"
echo "Full HMM/BSDS Validation Pipeline - Started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "============================================================"

# Load environment
source ~/.bashrc
conda activate unseam-validation 2>/dev/null || conda activate emofilm-hmm

cd ${SLURM_SUBMIT_DIR}/..
mkdir -p logs results/validation/{baldassano,yang,summary}

# Data paths
SHERLOCK_DATA=${SHERLOCK_DATA:-~/data/sherlock}
STUDYFORREST_DATA=${STUDYFORREST_DATA:-~/data/studyforrest}

echo ""
echo "Data Locations:"
echo "  Sherlock: ${SHERLOCK_DATA}"
echo "  StudyForrest: ${STUDYFORREST_DATA}"
echo ""

# Track overall status
VALIDATION_SUCCESS=true

# ============================================================
# Phase 1: Baldassano 2017 Replication
# ============================================================
echo ""
echo "============================================================"
echo "Phase 1: Baldassano 2017 Replication"
echo "============================================================"

if [ -d "${SHERLOCK_DATA}" ]; then
    echo "Running Baldassano validation..."

    python scripts/validate_baldassano.py \
        --data-dir ${SHERLOCK_DATA} \
        --roi AG \
        --n-events 25 \
        --method both \
        --output-dir results/validation/baldassano

    if [ $? -ne 0 ]; then
        echo "WARNING: Baldassano validation failed"
        VALIDATION_SUCCESS=false
    else
        echo "Baldassano validation completed successfully"
    fi
else
    echo "SKIPPED: Sherlock data not found at ${SHERLOCK_DATA}"
    VALIDATION_SUCCESS=false
fi

# ============================================================
# Phase 2: Yang 2023 Replication
# ============================================================
echo ""
echo "============================================================"
echo "Phase 2: Yang 2023 Replication"
echo "============================================================"

if [ -d "${STUDYFORREST_DATA}" ]; then
    echo "Running Yang validation..."

    python scripts/validate_yang.py \
        --data-dir ${STUDYFORREST_DATA} \
        --n-states 4 \
        --select-n-states \
        --cross-validate \
        --output-dir results/validation/yang

    if [ $? -ne 0 ]; then
        echo "WARNING: Yang validation failed"
        VALIDATION_SUCCESS=false
    else
        echo "Yang validation completed successfully"
    fi
else
    echo "SKIPPED: StudyForrest data not found at ${STUDYFORREST_DATA}"
    VALIDATION_SUCCESS=false
fi

# ============================================================
# Phase 3: Generate Summary Report
# ============================================================
echo ""
echo "============================================================"
echo "Phase 3: Generating Summary Report"
echo "============================================================"

# Create summary report
cat << 'EOF' > results/validation/summary/validation_report.md
# HMM/BSDS Validation Summary

## Validation Status

EOF

# Add Baldassano results if available
if [ -f "results/validation/baldassano/validation_summary.json" ]; then
    echo "### Baldassano 2017 Replication" >> results/validation/summary/validation_report.md
    echo "" >> results/validation/summary/validation_report.md
    echo '```json' >> results/validation/summary/validation_report.md
    cat results/validation/baldassano/validation_summary.json >> results/validation/summary/validation_report.md
    echo '```' >> results/validation/summary/validation_report.md
    echo "" >> results/validation/summary/validation_report.md
fi

# Add Yang results if available
if [ -f "results/validation/yang/yang_validation_summary.json" ]; then
    echo "### Yang 2023 Replication" >> results/validation/summary/validation_report.md
    echo "" >> results/validation/summary/validation_report.md
    echo '```json' >> results/validation/summary/validation_report.md
    cat results/validation/yang/yang_validation_summary.json >> results/validation/summary/validation_report.md
    echo '```' >> results/validation/summary/validation_report.md
fi

# Add timestamp
echo "" >> results/validation/summary/validation_report.md
echo "---" >> results/validation/summary/validation_report.md
echo "*Generated at: $(date)*" >> results/validation/summary/validation_report.md

echo "Summary report saved to: results/validation/summary/validation_report.md"

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "============================================================"
echo "VALIDATION PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Baldassano: results/validation/baldassano/"
echo "  - Yang: results/validation/yang/"
echo "  - Summary: results/validation/summary/"
echo ""

if [ "$VALIDATION_SUCCESS" = true ]; then
    echo "Status: ALL VALIDATIONS PASSED"
    exit 0
else
    echo "Status: SOME VALIDATIONS FAILED OR SKIPPED"
    echo "Check individual output directories for details"
    exit 1
fi
