# Event Segmentation Methods for fMRI Analysis

뇌 영상 데이터에서 event boundary를 탐지하기 위한 세 가지 방법론 구현.

## Overview

이 프로젝트는 naturalistic fMRI 데이터의 event segmentation을 위한 세 가지 접근법을 제공합니다:

| Method | 특징 | 논문 |
|--------|------|------|
| **BSDS** | Bayesian + AR dynamics + Factor model | Taghia et al. 2018 |
| **HMM-Baldassano** | Event-sequential HMM | Baldassano et al. 2017 |
| **HMM-Yang** | Standard GaussianHMM | Yang et al. 2023 |

## Quick Start

### 설치

```bash
pip install numpy scipy scikit-learn hmmlearn nilearn nibabel matplotlib
```

### 방법 비교 실행

```bash
# 시뮬레이션 데이터로 세 가지 방법 비교
python compare_methods.py --n-events 8

# Emo-Film 데이터로 BSDS 실행
python run_emofilm_bsds.py --task BigBuckBunny --n-states 8

# Emo-Film 데이터로 HMM 실행
python run_hmm_baseline.py --dataset emofilm --task BigBuckBunny --n-events 8
```

## Project Structure

```
BSDS_Project/
├── bsds_complete/           # BSDS Python 구현 (Taghia 2018)
│   ├── core/               # BSDSModel, BSDSConfig
│   ├── inference/          # HMM, Latent variable inference
│   ├── learning/           # Factor, AR, Transition learning
│   ├── analysis/           # Statistics, Visualization
│   └── utils/              # Math, Data utilities
│
├── hmm_baseline/            # HMM Baseline 구현
│   ├── model.py            # HMMEventSegment (Baldassano 스타일)
│   ├── hmmlearn_wrapper.py # HMMLearnWrapper (Yang 스타일)
│   ├── event_segmentation.py # Pure Python HMM
│   ├── comparison.py       # BSDS vs HMM 비교
│   └── data_loaders.py     # 데이터셋 로더
│
├── Taghia_Cai_NatureComm_2018-main/  # Original MATLAB (참조용)
├── docs/                    # 상세 문서
│   └── HMM_BASELINE_MANUAL.md
├── scripts/                 # SLURM job scripts
│   ├── run_comparison.slurm
│   └── run_hmm_emofilm.slurm
├── papers/                  # Reference papers (PDF)
│
├── run_emofilm_bsds.py     # BSDS CLI
├── run_hmm_baseline.py     # HMM CLI
├── compare_methods.py      # 방법 비교 스크립트
└── test_bsds_complete.py   # 테스트
```

## Python API

### BSDS

```python
from bsds_complete import BSDSModel, BSDSConfig

config = BSDSConfig(n_states=8, max_ldim=10, n_iter=100)
model = BSDSModel(config)
model.fit(data_list)  # List of (ROI x Time) arrays

states = model.get_states()
stats = model.get_summary_statistics()
```

### HMM-Baldassano (Event-Sequential)

```python
from hmm_baseline import HMMEventSegment, HMMConfig

config = HMMConfig(n_events=8, n_iter=100)
model = HMMEventSegment(config)
model.fit(data_list)  # List of (Time x Voxel) arrays

boundaries = model.get_event_boundaries(subject_idx=0)
```

### HMM-Yang (Standard GaussianHMM)

```python
from hmm_baseline.hmmlearn_wrapper import HMMLearnWrapper

model = HMMLearnWrapper(n_states=8, covariance_type='diag')
model.fit(data_list)

boundaries = model.get_event_boundaries(data_list[0])
```

## 비교 결과 예시

시뮬레이션 데이터 (150 timepoints, 6 events):

| Method | Avg F1 | Precision | Recall |
|--------|--------|-----------|--------|
| BSDS | 1.000 | 1.000 | 1.000 |
| HMM-Baldassano | 0.800 | 0.800 | 0.800 |
| HMM-Yang | 1.000 | 1.000 | 1.000 |

## 랩 서버 실행

```bash
# SLURM으로 job 제출
sbatch scripts/run_hmm_emofilm.slurm

# 또는 특정 task 지정
sbatch --export=TASK=FirstBite scripts/run_hmm_emofilm.slurm
```

자세한 내용: `scripts/README.md`

## Documentation

- **HMM Baseline 상세**: `docs/HMM_BASELINE_MANUAL.md`
- **Emo-Film 가이드**: `EMO_FILM_GUIDE.md`
- **BSDS 분석 리포트**: `ANALYSIS_REPORT.md`

## References

1. **Taghia et al. (2018)** - BSDS 원본
   - "Uncovering hidden brain state dynamics..."
   - *Nature Communications*, 9, 2505

2. **Baldassano et al. (2017)** - Event-sequential HMM
   - "Discovering event structure in continuous narrative..."
   - *Neuron*, 95(3), 709-721

3. **Yang et al. (2023)** - Standard GaussianHMM
   - "The default network dominates neural responses..."
   - *Nature Communications*, 14, 4400

## License

- MATLAB code: Original authors' license
- Python implementation: MIT License

---
*Last updated: 2026-01-14*
