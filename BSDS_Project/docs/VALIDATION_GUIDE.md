# HMM/BSDS Validation Guide

Method validation을 위한 단계별 가이드. Baldassano 2017과 Yang 2023 연구를 replicate하여 구현의 건전성을 확인합니다.

## 개요

### Validation 목표
1. **Baldassano et al. (2017) Replication**: Sherlock 데이터에서 HMM event boundary detection
2. **Yang et al. (2023) Replication**: StudyForrest 데이터에서 GaussianHMM 적용
3. **Ground Truth 비교**: Human-annotated boundaries와 모델 예측 비교

### 필요한 데이터셋

| Dataset | 용도 | Download |
|---------|------|----------|
| **Sherlock** | Baldassano replication | [Figshare](https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695) |
| **StudyForrest** | Yang replication | [Google Drive](https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t) |

---

## Step 1: 데이터 다운로드

### 1.1 Sherlock Dataset (Baldassano 2017)

```bash
# 로컬 또는 랩서버에서
mkdir -p ~/data/sherlock
cd ~/data/sherlock

# Figshare에서 다운로드 (약 500MB)
# 웹브라우저에서: https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695
# 또는 wget으로:
wget -O sherlock_atlas_data.zip "https://figshare.com/ndownloader/files/9021937"
unzip sherlock_atlas_data.zip
```

데이터 구조:
```
sherlock/
├── AG_movie_1TR.npy      # Angular Gyrus (Baldassano의 주요 ROI)
├── EAC_movie_1TR.npy     # Early Auditory Cortex
├── PMC_movie_1TR.npy     # Posterior Medial Cortex
└── ...
```

**핵심 정보:**
- TR = 1.5초
- 총 1976 TRs (약 48분)
- 17 subjects
- Human-annotated boundaries: 28개 (코드에 하드코딩됨)

### 1.2 StudyForrest Dataset (Yang 2023)

```bash
# Yang et al.의 전처리된 데이터
mkdir -p ~/data/studyforrest
cd ~/data/studyforrest

# Google Drive에서 수동 다운로드 필요:
# https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t

# 또는 gdown 사용:
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t
```

**핵심 정보:**
- TR = 2.0초
- Forrest Gump 영화 전체 (약 2시간)
- 15 subjects
- Yang et al.은 4개 state 사용

---

## Step 2: 랩서버 환경 설정

### 2.1 SSH 접속 및 환경 설정

```bash
# 랩서버 접속
ssh labserver

# 작업 디렉토리 설정
cd /your/workspace/
git clone https://github.com/kyungjinasusual/UNSEAM.git
cd UNSEAM/BSDS_Project

# Conda 환경 생성
conda create -n unseam-validation python=3.9
conda activate unseam-validation

# 의존성 설치
pip install numpy scipy scikit-learn hmmlearn
pip install matplotlib seaborn pandas
pip install nibabel nilearn  # fMRI 처리용
```

### 2.2 데이터 경로 설정

```bash
# 랩서버에서 데이터 경로 심볼릭 링크
ln -s /storage/bigdata/sherlock ~/data/sherlock
ln -s /storage/bigdata/studyforrest ~/data/studyforrest

# 또는 환경변수 설정
export SHERLOCK_DATA=/path/to/sherlock
export STUDYFORREST_DATA=/path/to/studyforrest
```

---

## Step 3: Baldassano 2017 Replication

### 3.1 실행

```bash
cd /path/to/UNSEAM/BSDS_Project

# Validation 스크립트 실행
python scripts/validate_baldassano.py \
    --data-dir ~/data/sherlock \
    --roi AG \
    --n-events 25 \
    --output-dir results/validation/baldassano
```

### 3.2 Expected Results

Baldassano 2017의 주요 발견:
- Angular Gyrus에서 HMM이 human boundaries와 가장 잘 일치
- ~25개 events가 최적
- Human-model boundary 일치도: ~35-40% (within ±3 TRs)

### 3.3 평가 메트릭

```python
from hmm_baseline.comparison import boundary_match_score

# Human boundaries (28개)
human_boundaries = SHERLOCK_HUMAN_BOUNDARIES

# Model boundaries
model_boundaries = model.get_event_boundaries(subject_idx=0)

# 평가 (tolerance = ±3 TRs)
scores = boundary_match_score(model_boundaries, human_boundaries, tolerance=3)
print(f"Precision: {scores['precision']:.3f}")
print(f"Recall: {scores['recall']:.3f}")
print(f"F1 Score: {scores['f1']:.3f}")
```

---

## Step 4: Yang 2023 Replication

### 4.1 실행

```bash
python scripts/validate_yang.py \
    --data-dir ~/data/studyforrest \
    --n-states 4 \
    --covariance-type diag \
    --output-dir results/validation/yang
```

### 4.2 Expected Results

Yang 2023의 주요 발견:
- Default Network가 movie story에 가장 민감
- 4개 state가 최적 (BIC 기반)
- Cross-validation log-likelihood로 모델 비교

---

## Step 5: SLURM Job 제출

### 5.1 Baldassano Validation Job

```bash
# scripts/validate_baldassano.slurm
sbatch scripts/validate_baldassano.slurm
```

### 5.2 전체 Validation Job

```bash
# 모든 validation을 한번에
sbatch scripts/run_full_validation.slurm
```

### 5.3 Job 모니터링

```bash
# Job 상태 확인
squeue -u $USER

# 실시간 로그 확인
tail -f logs/validation_*.out

# 결과 확인
ls -la results/validation/
```

---

## Step 6: 결과 시각화

### 6.1 Boundary Comparison Plot

```python
from hmm_baseline.comparison import plot_boundary_comparison

plot_boundary_comparison(
    model_boundaries=model_boundaries,
    human_boundaries=human_boundaries,
    n_timepoints=1976,
    TR=1.5,
    output_file='results/validation/boundary_comparison.png'
)
```

### 6.2 State Sequence Plot

```python
from hmm_baseline.comparison import plot_state_sequence

plot_state_sequence(
    states=states,
    boundaries=model_boundaries,
    human_boundaries=human_boundaries,
    output_file='results/validation/state_sequence.png'
)
```

---

## 성공 기준

### Baldassano Replication
- [ ] Human-model boundary 일치: F1 > 0.30 (within ±3 TRs)
- [ ] AG에서 가장 좋은 일치
- [ ] ~25 events가 최적

### Yang Replication
- [ ] 4-state model이 BIC 기준 최적
- [ ] Cross-validation LL > random baseline
- [ ] Default network regions에서 긴 event duration

### Code Sanity Check
- [ ] Simulated data에서 F1 = 1.0
- [ ] 수렴 안정성 확인
- [ ] 다른 random seed에서도 유사한 결과

---

## Troubleshooting

### 데이터 로드 오류
```bash
# npy 파일 형태 확인
python -c "import numpy as np; d = np.load('AG_movie_1TR.npy'); print(d.shape)"
```

### 메모리 부족
```bash
# SLURM에서 메모리 늘리기
sbatch --mem=64G scripts/validate_baldassano.slurm
```

### 수렴 문제
```python
# n_iter 늘리기
model = HMMEventSegment(config)
config.n_iter = 200
config.tol = 1e-5
```

---

## 참고 자료

### 원본 코드
- Baldassano: https://github.com/cbaldassano/Event-Segmentation
- BrainIAK: https://brainiak.org/tutorials/12-hmm/
- Yang: https://github.com/dblabs-mcgill-mila/hmm_forrest_nlp

### 논문
- Baldassano et al. (2017) Neuron
- Yang et al. (2023) Nature Communications
- Taghia et al. (2018) Nature Communications

---

*Last updated: 2026-01-14*
