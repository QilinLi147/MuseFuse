# MuseFuse: Multi-View Feature Fusion for Music Emotion Recognition

Official implementation of **MuseFuse** framework for music emotion recognition on **Memo2496** dataset, as submitted in *IEEE Transactions on Affective Computing*.

## Framework Overview

MuseFuse integrates three synergistic modules to address multi-view music emotion recognition challenges:

- **ProtoAlign**: Prototype-guided semantic alignment with exponential moving average updates (momentum = 0.9)
- **ReliaPseudo**: Reliability-guided self-training with entropy-based weighting and curriculum thresholding (0.6→0.3)
- **TriDistill**: Symmetric tri-branch knowledge distillation across fusion-Mel-cochleagram branches (T=2.0)

## Requirements

- Python 3.8+
- PyTorch 1.13.0+ with CUDA 12.1

## Dataset Preparation

### Download Memo2496

The Memo2496 dataset contains 2496 expert-annotated instrumental music tracks with continuous Valence and Arousal labels.

**Download Links**:

- Figshare: https://figshare.com/articles/dataset/Memo2496/25827034
- IEEE DataPort: https://dx.doi.org/10.21227/3824-wy49

### Expected Directory Structure

Place the downloaded files in the following structure:

```text
/data/qilin.li/dataset/memo/
├── mel_spec.npy          # Mel-spectrogram features (2496, 128, 87)
├── cochlegram.npy        # Cochleagram features (2496, 84, 87)
└── labels/
    ├── label_a.npy       # Arousal labels (2496,)
    └── label_v.npy       # Valence labels (2496,)
```

**Note**: Update the `--data_root` argument if your dataset is stored in a different location.

## Quick Start

### Train on Arousal Dimension

```bash
python train_musefuse.py --label_mode a
```

### Train on Valence Dimension

```bash
python train_musefuse.py --label_mode v
```

### Custom Configuration

```bash
python train_musefuse.py \
    --label_mode a \
    --epochs 80 \
    --batch_size 256 \
    --lr 1e-3 \
    --data_root /your/custom/path \
    --out_dir ./checkpoints \
    --mixed_precision
```

## Expected Results

Performance on Memo2496 dataset (as reported in the paper):

| Dimension | Accuracy | F1    | AUC   |
| :-------- | :------- | :---- | :---- |
| Arousal   | 83.40%   | 81.05 | 91.01 |
| Valence   | 79.32%   | 85.27 | 85.09 |

## Model Checkpoints

Trained checkpoints are saved in `checkpoints_musefuse/` with naming format:

```text
musefuse_memo_{a|v}_{timestamp}.pt
```

Each checkpoint contains:

- Model state dict
- Training configuration
- Validation metrics

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

This work was supported by:

- National Natural Science Foundation of China (Grant No. 62222603)
- STI2030-Major Projects (2021ZD0200700)
- Key-Area R&D Programme of Guangdong Province (2023B0303030001)
- Guangdong Introducing Innovative and Entrepreneurial Teams (2019ZT08X214)
- Science and Technology Programme of Guangzhou (2024A04J6310)

