# Imitation Learning Scripts

LeRobot을 사용한 Diffusion Policy 학습 예제

## 📋 요구사항

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA 12.1+ 호환 GPU
- 최소 8GB GPU 메모리

## 🚀 Quick Start

### 1. Conda 환경 생성 및 활성화

```bash
# Conda 환경 생성 (처음 한 번만)
conda env create -f environment.yml

# 환경 활성화
conda activate imitation-learning

# Pre-commit hook 설정 (선택사항)
pre-commit install
```

### 2. 학습 실행

```bash
python train.py
```

학습된 모델은 `outputs/train/example_pusht_diffusion/`에 저장됩니다.

## 🎯 Makefile 명령어

편리한 명령어들:

```bash
make setup          # Conda 환경 생성 + pre-commit 설정
make train          # 학습 시작
make format         # 코드 포맷팅 (black, isort)
make lint           # 코드 검사 (flake8)
make check          # 포맷팅 + 검사
make clean          # 출력 파일 정리
make clean-env      # Conda 환경 삭제
make update-env     # 환경 업데이트
make help           # 모든 명령어 보기
```

## 🔄 환경 관리

```bash
# 환경 업데이트
conda env update -f environment.yml --prune
# 또는
make update-env

# 환경 삭제
conda env remove -n imitation-learning
# 또는
make clean-env

# 환경 목록 확인
conda env list

# 설치된 패키지 확인
conda list
```

## 📂 프로젝트 구조

```
.
├── train.py                # 학습 스크립트
├── environment.yml         # Conda 환경
├── pyproject.toml          # Black, isort 설정
├── .flake8                 # Flake8 설정
├── .pre-commit-config.yaml # Pre-commit hooks 설정
├── Makefile                # 편리한 명령어 모음
├── outputs/                # 학습 결과물 (자동 생성)
└── ...
```

## 💻 개발 워크플로우

```bash
# 1. 환경 설정 (처음 한 번)
make setup

# 2. 환경 활성화
conda activate imitation-learning

# 3. 코드 작업
# ... 코드 수정 ...

# 4. 포맷팅 + 검사
make check

# 5. Git 커밋 (pre-commit이 자동으로 검사)
git add .
git commit -m "Add feature"

# 6. 학습 실행
make train
```

## 🐛 문제 해결

### CUDA 버전 불일치

```bash
# 시스템의 CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"
```

다른 CUDA 버전이 필요한 경우 `environment.yml`에서 `pytorch-cuda` 수정:

```yaml
# CUDA 11.8의 경우
- pytorch-cuda=11.8

# CUDA 12.1의 경우 (기본값)
- pytorch-cuda=12.1
```

### 환경이 꼬인 경우

```bash
# 환경 완전 삭제 후 재생성
conda env remove -n imitation-learning
conda env create -f environment.yml
```

### Conda가 느린 경우

Mamba를 사용하면 훨씬 빠릅니다:

```bash
# Mamba 설치 (한 번만)
conda install -n base conda-forge::mamba

# Mamba로 환경 생성
mamba env create -f environment.yml

# Mamba로 환경 업데이트
mamba env update -f environment.yml --prune
```

## 💡 팁

### 빠른 시작

```bash
# 한 줄로 환경 생성 + 학습
make setup && conda activate imitation-learning && make train
```

### GPU 메모리 부족

`train.py`에서 배치 크기 줄이기:

```python
batch_size=32  # 기본값 64에서 감소
```

## 📝 License

Apache License 2.0
