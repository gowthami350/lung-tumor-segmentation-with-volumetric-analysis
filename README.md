# Lung Cancer Segmentation

Deep learning-based lung cancer segmentation using U-Net++ architecture.

## Overview

Multi-class segmentation of lung cancer types (ADC, LCC, SCC) from CT images using U-Net++ with EfficientNet-B4 encoder.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python training/train.py
```

## Data Structure Required

```
backend/data/raw/
├── train/
│   ├── CT/
│   │   ├── ADC/     # CT images
│   │   ├── LCC/
│   │   └── SCC/
│   └── MASK/
│       ├── ADC/     # Masks
│       ├── LCC/
│       └── SCC/
└── test/            # Same structure
```

## What's Included

- ✅ U-Net++ training code
- ✅ Requirements file
- ❌ Pre-trained models (train your own)
- ❌ Dataset (provide your own)

## Results

- **Model**: U-Net++ with EfficientNet-B4
- **Performance**: 80-90% Dice Score
- **Classes**: Background, ADC, LCC, SCC

## Requirements

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM
- GPU recommended

## License

Research Use Only