# SLURM Job Scripts

Lab server에서 실행하기 위한 SLURM 스크립트들.

## 워크플로우

```
로컬 (Desktop)              랩서버
─────────────────────────────────────────────────
1. 코드 작성/수정
2. git commit & push  ───▶  3. git pull
                            4. sbatch 스크립트 실행
                            5. 결과 생성
                            6. git add/commit/push
7. git pull ◀────────────────
8. 결과 확인
```

## 사용법

### 서버에서 저장소 클론
```bash
ssh labserver
cd /your/workspace/
git clone https://github.com/kyungjinasusual/UNSEAM.git
cd UNSEAM/BSDS_Project
```

### 환경 설정
```bash
conda create -n emofilm-hmm python=3.9
conda activate emofilm-hmm
pip install numpy scipy scikit-learn hmmlearn
pip install nibabel nilearn matplotlib seaborn
```

### Job 제출

#### 시뮬레이션 데이터로 테스트
```bash
sbatch scripts/run_comparison.slurm
```

#### Emo-Film 데이터로 HMM 실행
```bash
# 기본 (BigBuckBunny)
sbatch scripts/run_hmm_emofilm.slurm

# 다른 task 지정
sbatch --export=TASK=FirstBite,N_EVENTS=10 scripts/run_hmm_emofilm.slurm
```

### Job 상태 확인
```bash
# 내 job 목록
squeue -u $USER

# 특정 job 상태
squeue -j <JOB_ID>

# job 취소
scancel <JOB_ID>

# 출력 확인 (실시간)
tail -f logs/hmm_emofilm_<JOB_ID>.out
```

### 결과 확인
```bash
# 결과 파일 목록
ls -la results/

# 결과를 git에 커밋
git add results/
git commit -m "Add HMM analysis results"
git push
```

## 스크립트 설명

| 스크립트 | 용도 | 시간 | 메모리 |
|---------|------|------|--------|
| `run_comparison.slurm` | BSDS vs HMM 비교 (전체) | 4h | 32GB |
| `run_hmm_emofilm.slurm` | HMM만 Emo-Film에 실행 | 2h | 16GB |

## GPU 필요 없음

현재 구현된 방법들(BSDS, HMM)은 **CPU 기반**입니다.
- BSDS: Variational inference (CPU intensive)
- HMM: Forward-backward, EM algorithm (CPU)

GPU partition 대신 `normal` partition 사용 권장.

## 문제 해결

### "ModuleNotFoundError: No module named 'hmm_baseline'"
```bash
# BSDS_Project 디렉토리에서 실행 확인
cd /path/to/UNSEAM/BSDS_Project
python -c "from hmm_baseline import HMMEventSegment"
```

### "FileNotFoundError: Emo-FiLM data not found"
```bash
# 데이터 경로 확인
ls /storage/bigdata/Emo-FiLM/
# 경로가 다르면 스크립트의 EMOFILM_DATA 변수 수정
```

### 메모리 부족
```bash
# 메모리 늘려서 재제출
sbatch --mem=64G scripts/run_comparison.slurm
```
