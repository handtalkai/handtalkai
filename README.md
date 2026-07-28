# 🖐️ Real-Time Hand Gesture Recognition AI System

A production-grade, modular gesture recognition system using **MediaPipe Hands**, **TensorFlow/Keras**, and **OpenCV**. Supports static and dynamic (LSTM) gesture classification with real-time webcam inference.

---

## 📦 Features

| Feature | Details |
|---|---|
| **Landmark Extraction** | MediaPipe 21-point hand skeleton, wrist-normalized |
| **Static Model** | Dense residual network + BatchNorm + Focal Loss |
| **LSTM Model** | Bidirectional LSTM + self-attention for dynamic gestures |
| **Ensemble** | Weighted static + LSTM prediction fusion |
| **Augmentation** | Jitter, rotation, scaling → 3× dataset size |
| **Class Balance** | Computed class weights + Focal Loss |
| **Smoothing** | Rolling majority-vote window (10 frames) |
| **Low-Light** | CLAHE preprocessing per frame |
| **TTS** | Non-blocking pyttsx3 speech output |
| **Reports** | Accuracy curves, confusion matrix, JSON report |

---

## 🧩 Project Structure

```
gesture_ai/
├── main.py                      # CLI entry point
├── train_pipeline.py            # End-to-end training script
├── generate_synthetic_data.py   # Synthetic dataset generator
├── requirements.txt
├── src/
│   ├── config.py                # All hyperparameters & paths
│   ├── landmark_extractor.py    # MediaPipe wrapper + normalization
│   ├── dataset.py               # Recording, augmentation, loading
│   ├── models.py                # Static & LSTM architectures
│   ├── trainer.py               # Training + evaluation + reports
│   ├── predictor.py             # StaticPredictor / LSTMPredictor / Ensemble
│   ├── ui.py                    # OpenCV overlay rendering
│   └── speech.py                # TTS output
├── data/
│   └── processed/landmarks.csv  # Landmark dataset
├── models/
│   ├── gesture_static.keras     # Trained static model
│   └── label_encoder.npy        # Class label mapping
└── reports/
    ├── history_static.png        # Training curves
    ├── confusion_static.png      # Confusion matrix
    └── report_static.json        # Full evaluation report
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2A. Use synthetic data + train immediately
```bash
python train_pipeline.py --synthetic --samples 300
```

### 2B. Collect real webcam data
```bash
# Repeat for each gesture: Hello, Hi, Stop, Thank You, Yes, No
python main.py collect Hello --samples 200
python main.py collect Hi    --samples 200
python main.py collect Stop  --samples 200
# ...

# Then train:
python main.py train
```

### 3. Run real-time recognition
```bash
python main.py demo                    # Static model
python main.py demo --ensemble         # Static + LSTM ensemble
python main.py demo --tts              # With speech output
python main.py demo --camera 1         # Different camera
```

---

## 🧠 Model Architecture

### StaticGestureNet
```
Input (63,) → Dense(256) → BN → ReLU → Dropout(0.4)
           → Dense(256) → BN → ReLU → Dropout(0.4) + Residual
           → Dense(128) → BN → ReLU → Dropout(0.3)
           → Dense(64)  → BN → ReLU → Dropout(0.2)
           → Dense(6, softmax)
```
Loss: **Focal Loss** (γ=2, α=0.25) — reduces impact of easy samples

### LSTMGestureNet
```
Input (30, 63) → BiLSTM(128, return_seq=True) + Dropout(0.3)
              → BiLSTM(64,  return_seq=True) + Dropout(0.3)
              → Self-Attention → Sum → Dense(128) → Dense(6, softmax)
```

---

## 🎯 Supported Gestures

| Gesture | Description |
|---|---|
| **Hello** | Full open hand, waving motion |
| **Hi** | Four fingers open (compact wave) |
| **Stop** | Flat palm facing forward, all fingers extended |
| **Thank You** | Open hand angled toward face |
| **Yes** | Thumbs up (fist + thumb open) |
| **No** | Index finger pointing / wagging |

---

## ⌨️ Demo Keyboard Controls

| Key | Action |
|---|---|
| `L` | Toggle live inference |
| `S` | Toggle hand skeleton overlay |
| `M` | Cycle model (Static ↔ Ensemble) |
| `H` | Toggle help overlay |
| `Q` | Quit |

---

## 📊 Performance (Synthetic Data)

| Metric | Value |
|---|---|
| Test Accuracy | **91.4%** |
| Stop F1-Score | 0.99 |
| Yes F1-Score | 1.00 |
| No F1-Score | 1.00 |
| Epochs to converge | 10–25 |

> Note: With real webcam data, expect 95–99% accuracy for well-separated gestures.

---

## 🔧 Configuration

All hyperparameters are in `src/config.py`:

```python
CONFIDENCE_THRESHOLD = 0.80    # Below this → "Unknown Gesture"
SMOOTHING_WINDOW     = 10      # Frames for majority-vote smoothing
SEQUENCE_LENGTH      = 30      # LSTM input frames
AUGMENTATION_FACTOR  = 3       # Dataset multiplication
DROPOUT_RATE         = 0.4
LEARNING_RATE        = 0.001
BATCH_SIZE           = 32
EPOCHS               = 100
PATIENCE             = 15      # Early stopping
```

---

## 🚀 Advanced: LSTM Training

```bash
python train_pipeline.py --synthetic --lstm
# or
python main.py train --lstm
```

Then run ensemble inference:
```bash
python main.py demo --ensemble
```

---

## 📈 Reports

After training, the `reports/` folder contains:
- `history_static.png` — accuracy & loss curves
- `confusion_static.png` — per-class confusion matrix (raw + normalised)
- `class_distribution.png` — sample counts per class
- `report_static.json` — full classification report (precision/recall/F1)
