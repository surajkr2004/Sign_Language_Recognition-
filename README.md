# Sign Language Recognition — CNN-based ASL Alphabet Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red.svg)](https://mediapipe.dev)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.9%25-brightgreen.svg)]()

> I built this app to translate live American Sign Language into text through a webcam! I trained my own custom AI model (CNN) from scratch using TensorFlow and hooked it up to OpenCV and MediaPipe to track hand movements in real-time.

---

## Demo

```
[Webcam feed]  →  MediaPipe hand detection  →  CNN prediction  →  Letter/Word output
```

Controls during live demo:
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save current frame |
| `C` | Clear text buffer |

---

## Project Structure

```
sign language/
├── data/
│   ├── raw/                    ← Place Kaggle CSVs here
│   ├── processed/
│   └── dataset_info.md
├── models/
│   ├── sign_language_cnn.h5    ← Saved after training
│   └── checkpoints/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_visualization.ipynb
├── src/
│   ├── config.py               ← All hyperparameters & paths
│   ├── data_loader.py          ← Dataset + augmentation pipeline
│   ├── model.py                ← CNN architecture (Keras functional API)
│   ├── train.py                ← Full training pipeline (CLI runnable)
│   ├── evaluate.py             ← Metrics, confusion matrix, sample preds
│   └── inference.py            ← Real-time webcam demo
├── results/                    ← Generated plots & reports
├── requirements.txt
└── README.md
```

---

## CNN Architecture

```
Input (64×64×1 grayscale)
    ↓
Conv2D(32)  + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
    ↓
Conv2D(64)  + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
    ↓
Conv2D(128) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
    ↓
Conv2D(256) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.30)
    ↓
Flatten → Dense(512, ReLU) + Dropout(0.50)
    ↓
Dense(24, Softmax)
```

**Optimizer**: Adam (lr=0.001 with ReduceLROnPlateau)  
**Loss**: Categorical Cross-Entropy  
**Regularization**: L2 + Dropout + BatchNorm  
**Callbacks**: EarlyStopping (patience=8), ModelCheckpoint, ReduceLROnPlateau  

---

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Download and extract
3. Place files as:
   - `data/raw/sign_mnist_train.csv`
   - `data/raw/sign_mnist_test.csv`

### 3. Validate Pipeline (dry run)

```bash
python src/train.py --dry-run
```

### 4. Train

```bash
python src/train.py
# or with custom epochs:
python src/train.py --epochs 30
```

### 5. Evaluate

```bash
python src/evaluate.py
```

Generates:
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/sample_predictions.png`

### 6. Live Demo

```bash
python src/inference.py
# or with a specific camera:
python src/inference.py --camera 0
```

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **99.98%** |
| Parameters | ~2.1M |
| Training time | ~20 min (Early stopping at 22 epochs) |
| Inference speed | ~30-60 FPS |

---

## Dataset

**Sign Language MNIST** — Kaggle  
27,455 training images · 7,172 test images · 24 classes  
Grayscale 28×28 pixel images (resized to 64×64 for this model)

---

## Technical Stack

| Component | Library |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Hand Detection | MediaPipe Hands |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter |

---

## License

MIT License — free to use, modify, and distribute.
