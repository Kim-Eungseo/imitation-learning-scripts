# Imitation Learning Scripts

LeRobot 기반 Imitation Learning 학습 및 평가 (Diffusion, ACT, TDMPC)

## 🚀 Quick Start

```bash
# 환경 설정
conda env create -f environment.yml
conda activate imitation-learning

# 학습
python train.py                    # Diffusion (기본)
python train.py --policy act       # ACT
python train.py --policy tdmpc     # TDMPC

# 평가
python eval.py --pretrained_path outputs/train/lerobot_pusht_diffusion
python eval.py --pretrained_path outputs/train/lerobot_pusht_diffusion --render
```

## 🎯 Makefile

```bash
make train          # Diffusion 학습
make train-act      # ACT 학습
make train-tdmpc    # TDMPC 학습
make eval           # 모델 평가
make eval-render    # 시각화와 함께 평가
make update-pip     # pip 의존성 업데이트
make help           # 전체 명령어
```

## 📝 License

Apache License 2.0
