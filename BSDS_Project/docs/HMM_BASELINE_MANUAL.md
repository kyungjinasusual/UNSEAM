# HMM Baseline for Event Segmentation - 상세 매뉴얼

## 개요

이 모듈은 fMRI 시계열 데이터에서 event boundary를 탐지하기 위한 HMM (Hidden Markov Model) 기반 baseline 구현입니다. 두 가지 접근법을 제공합니다:

1. **Baldassano 2017 스타일** (Event-Sequential HMM)
2. **Yang 2023 스타일** (Standard GaussianHMM)

---

## 참조 논문 및 코드

### 1. Baldassano et al. 2017 (Neuron)

| 항목 | 내용 |
|------|------|
| **논문** | "Discovering Event Structure in Continuous Narrative Perception and Memory" |
| **링크** | https://pubmed.ncbi.nlm.nih.gov/28772125/ |
| **원본 코드** | https://github.com/cbaldassano/Event-Segmentation |
| **BrainIAK 구현** | https://github.com/brainiak/brainiak/tree/master/brainiak/eventseg |
| **튜토리얼** | https://brainiak.org/tutorials/12-hmm/ |

**핵심 특징:**
- Event-sequential constraint: 상태 k에서는 k 또는 k+1로만 전이 가능
- Group-level 분석에 최적화
- Gaussian emission model with shared variance

### 2. Yang et al. 2023 (Nature Communications)

| 항목 | 내용 |
|------|------|
| **논문** | "The Default Network Dominates Neural Responses to Evolving Movie Stories" |
| **링크** | https://www.nature.com/articles/s41467-023-39862-y |
| **원본 코드** | https://github.com/dblabs-mcgill-mila/hmm_forrest_nlp |
| **데이터** | https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t |

**핵심 특징:**
- Standard HMM: 임의의 상태 전이 허용
- Single-subject level 분석 가능
- hmmlearn 라이브러리 활용
- 최적 상태 수 선택을 위한 BIC/cross-validation

---

## 원본과의 비교

### Baldassano 스타일 (`HMMEventSegment`)

| 구성요소 | 원본 (BrainIAK) | 우리 구현 | 차이점 |
|----------|-----------------|-----------|--------|
| Forward-Backward | ✅ | ✅ | 동일한 알고리즘 |
| Viterbi decoding | ✅ | ✅ | 동일 |
| Event-sequential 제약 | ✅ | ✅ | 동일 |
| Gaussian emission | ✅ | ✅ | 동일 |
| Multi-subject fitting | ✅ | ✅ | 동일 |
| BrainIAK 의존성 | 필수 | 선택적 | **우리 구현은 순수 Python 대체 제공** |
| Log-space 연산 | ✅ | ✅ | 수치 안정성 동일 |

**추가 기능:**
- BrainIAK 없이도 작동하는 순수 Python 구현
- BSDS와 직접 비교 가능한 인터페이스
- 저장/로드 기능

### Yang 스타일 (`HMMLearnWrapper`)

| 구성요소 | 원본 (hmm_forrest_nlp) | 우리 구현 | 차이점 |
|----------|------------------------|-----------|--------|
| GaussianHMM | hmmlearn | hmmlearn | 동일 |
| Covariance type | 'full' | 선택 가능 | **우리는 'diag', 'full' 모두 지원** |
| n_states | 4 (고정) | 자유 선택 | **우리는 유연하게 지정** |
| n_iter | 500 | 선택 가능 | 동일 |
| n_init | 100 | 선택 가능 | 동일 |
| Cross-validation | 20-fold | 지원 | 동일 |
| BIC model selection | ✅ | ✅ | 동일 |

**추가 기능:**
- `select_optimal_n_states()` 함수로 자동 상태 수 선택
- BSDS/Baldassano HMM과 동일한 출력 형식
- Boundary 추출 유틸리티

---

## 모듈 구조

```
hmm_baseline/
├── __init__.py              # 패키지 초기화, 주요 클래스 export
├── config.py                # HMMConfig 데이터클래스
├── model.py                 # HMMEventSegment (Baldassano 스타일)
├── hmmlearn_wrapper.py      # HMMLearnWrapper (Yang 스타일)
├── event_segmentation.py    # 순수 Python HMM 구현
├── data_loaders.py          # 데이터셋 로더 (Sherlock, Emo-Film 등)
└── comparison.py            # BSDS vs HMM 비교 유틸리티
```

---

## 사용법

### 기본 사용 (Baldassano 스타일)

```python
from hmm_baseline import HMMEventSegment, HMMConfig

# 설정
config = HMMConfig(
    n_events=8,        # 이벤트 (상태) 수
    n_iter=100,        # EM 반복 횟수
    tol=1e-4,          # 수렴 기준
    TR=2.0,            # Repetition Time (초)
    random_seed=42
)

# 모델 생성 및 학습
# data_list: List of (T, V) arrays - T=timepoints, V=voxels/ROIs
model = HMMEventSegment(config)
model.fit(data_list, use_brainiak=False)  # Python 구현 사용

# 결과 추출
boundaries = model.get_event_boundaries(subject_idx=0)  # TR 인덱스
timestamps = model.get_boundaries_timestamp(subject_idx=0)  # 초 단위
occupancy = model.compute_occupancy(subject_idx=0)  # 상태별 점유율

print(f"Event boundaries (TR): {boundaries}")
print(f"Event boundaries (sec): {timestamps}")
```

### Yang 스타일 (Standard HMM)

```python
from hmm_baseline.hmmlearn_wrapper import HMMLearnWrapper, select_optimal_n_states

# 최적 상태 수 선택 (선택적)
optimal_k, scores = select_optimal_n_states(
    data_list,
    k_range=(3, 10),
    criterion='bic'
)
print(f"Optimal n_states: {optimal_k}")

# 모델 학습
model = HMMLearnWrapper(
    n_states=optimal_k,
    covariance_type='diag',  # 또는 'full'
    n_iter=100,
    n_init=10,
    random_seed=42
)
model.fit(data_list)

# 결과 추출 (데이터 직접 전달)
boundaries = model.get_event_boundaries(data_list[0])
occupancy = model.compute_occupancy(data_list[0])
```

### CLI 사용

```bash
# 시뮬레이션 데이터로 테스트
python run_hmm_baseline.py --mode test

# Emo-Film 데이터로 실행
python run_hmm_baseline.py \
    --dataset emofilm \
    --task BigBuckBunny \
    --n-events 8 \
    --output-dir results/hmm_baseline

# BSDS와 비교
python compare_methods.py --n-events 8
```

---

## BSDS와의 비교

### 개념적 차이

| 측면 | BSDS | HMM-Baldassano | HMM-Yang |
|------|------|----------------|----------|
| **상태 전이** | 자유 | Sequential only | 자유 |
| **Emission** | Factor model + AR(1) | Gaussian | Gaussian |
| **잠재 변수** | Latent factors | 없음 | 없음 |
| **시간 역학** | AR(1) dynamics | 없음 | 없음 |
| **Bayesian** | ✅ Variational | ❌ MLE | ❌ MLE |
| **복잡도** | 높음 | 중간 | 낮음 |
| **해석력** | 높음 | 중간 | 중간 |

### 언제 무엇을 사용하나?

**BSDS 권장:**
- 뇌 상태의 dynamic patterns 분석 필요
- Latent factor 해석 필요
- 시간적 자기상관 모델링 필요
- 연산 자원이 충분할 때

**HMM-Baldassano 권장:**
- Event boundary 위치만 필요
- 이벤트가 순차적으로 발생하는 것이 확실할 때
- Group-level 분석
- 빠른 baseline 필요

**HMM-Yang 권장:**
- 상태 전이가 자유로운 경우
- Single-subject 분석
- BIC로 최적 상태 수 선택 필요
- hmmlearn 생태계 활용

---

## 출력 형식

### Event Boundaries

```python
# TR 인덱스 (0-based)
boundaries_tr = [26, 45, 72, 98, 125]

# 타임스탬프 (초)
boundaries_sec = [52.0, 90.0, 144.0, 196.0, 250.0]  # TR=2.0초 가정
```

### Sherlock Human Boundaries (참조용)

```python
# Baldassano 2017에서 사용된 human-annotated boundaries
SHERLOCK_HUMAN_BOUNDARIES = [
    26, 35, 56, 72, 86, 108, 131, 143, 157, 173,
    192, 204, 226, 313, 362, 398, 505, 526, 533,
    568, 616, 634, 678, 696, 747, 780, 870, 890
]
# 총 28개 경계, TR=1.5초
```

---

## 평가 메트릭

### Boundary Match Score

```python
from hmm_baseline.comparison import boundary_match_score

scores = boundary_match_score(
    predicted_boundaries,
    true_boundaries,
    tolerance=3  # ±3 TR 이내면 매칭
)

print(f"Precision: {scores['precision']:.3f}")
print(f"Recall: {scores['recall']:.3f}")
print(f"F1 Score: {scores['f1']:.3f}")
```

### 비교 리포트

```python
from hmm_baseline.comparison import generate_comparison_report

report = generate_comparison_report(
    hmm_model,
    bsds_model,
    subject_idx=0,
    human_boundaries=SHERLOCK_HUMAN_BOUNDARIES
)
print(report)
```

---

## 구현 세부사항

### Forward-Backward Algorithm

```
Forward Pass (α):
  α[0, 0] = 1.0 (첫 상태에서 시작)
  α[t, k] = α[t-1, k] * P(stay) * P(obs) + α[t-1, k-1] * P(trans) * P(obs)

Backward Pass (β):
  β[T-1, K-1] = 1.0 (마지막 상태에서 종료)
  β[t, k] = β[t+1, k] * P(stay) * P(obs) + β[t+1, k+1] * P(trans) * P(obs)

Posterior:
  γ[t, k] = α[t, k] * β[t, k] / Σ_k(α[t, k] * β[t, k])
```

### Log-space 연산

수치 안정성을 위해 모든 확률 계산은 log-space에서 수행:

```python
def log_sum_exp(log_a, log_b):
    """Numerically stable log(exp(a) + exp(b))"""
    if log_a > log_b:
        return log_a + np.log1p(np.exp(log_b - log_a))
    else:
        return log_b + np.log1p(np.exp(log_a - log_b))
```

---

## 참고 문헌

1. Baldassano, C., Chen, J., Zadbood, A., Pillow, J. W., Hasson, U., & Norman, K. A. (2017). Discovering event structure in continuous narrative perception and memory. *Neuron*, 95(3), 709-721.

2. Yang, E., Kim, J., & Chen, J. (2023). The default network dominates neural responses to evolving movie stories. *Nature Communications*, 14, 4400.

3. Taghia, J., Cai, W., Ryali, S., Kochalka, J., Nicholas, J., Chen, T., & Menon, V. (2018). Uncovering hidden brain state dynamics that regulate performance and decision-making during cognition. *Nature Communications*, 9, 2505.

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|-----------|
| 2026-01-14 | 1.0.0 | 초기 구현 - Baldassano & Yang 스타일 HMM |

---

*작성: UNSEAM Project*
*최종 수정: 2026-01-14*
