# Imitation Learning Scripts

LeRobot 기반 Imitation Learning 학습 (Diffusion, ACT, TDMPC)

## 🚀 Quick Start

```bash
# 환경 생성 및 활성화
conda env create -f environment.yml
conda activate imitation-learning

# 학습 실행
python train.py                    # Diffusion (기본)
python train.py --policy act       # ACT
python train.py --policy tdmpc     # TDMPC
```

## 🤖 지원 정책

| 정책 | 설명 | 명령어 |
|------|------|--------|
| **Diffusion** | 노이즈 제거 기반 행동 생성 | `--policy diffusion` |
| **ACT** | Transformer 기반 행동 청킹 | `--policy act` |
| **TDMPC** | TD Learning + MPC | `--policy tdmpc` |

## ⚙️ 학습 파라미터

```bash
python train.py --help  # 전체 옵션 확인
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--policy` | `diffusion` | 정책 (diffusion, act, tdmpc) |
| `--dataset_name` | `lerobot/pusht` | 데이터셋 |
| `--training_steps` | `5000` | 학습 스텝 |
| `--batch_size` | `64` | 배치 크기 |
| `--learning_rate` | `1e-4` | 학습률 |

### 예시

```bash
python train.py --policy act --training_steps 10000 --batch_size 32
python train.py --policy tdmpc --dataset_name lerobot/aloha_sim_insertion_human
```

## 🎯 Makefile

```bash
make train          # Diffusion 학습
make train-act      # ACT 학습
make train-tdmpc    # TDMPC 학습
make help           # 전체 명령어
```

## 🐛 문제 해결

### 환경 재설정
```bash
conda env remove -n imitation-learning
conda env create -f environment.yml
```

### GPU 메모리 부족
```bash
python train.py --batch_size 32
```

## 📝 License

Apache License 2.0
