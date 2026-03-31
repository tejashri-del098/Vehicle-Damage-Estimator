# 🚗 Vehicle Damage Estimator

> **AI-powered vehicle damage detection and repair cost estimation using YOLOv8 Instance Segmentation + Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics_8.1.29-00FFAA?style=flat-square)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.24.3-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)

---

## 📌 Overview

**Vehicle Damage Estimator** is an end-to-end computer vision pipeline that takes an image of a damaged vehicle and outputs an instant repair cost estimate. It uses **YOLOv8 Instance Segmentation** to produce pixel-dense masks of damaged regions, then maps those masks to severity levels and industry-average repair cost ranges.

The system supports both a browser-based **Streamlit web UI** and a **command-line interface (CLI)** for flexible deployment.

---

## ✨ Features

- 🔍 **Instance Segmentation** — Uses YOLOv8-seg to isolate damaged pixels with mask precision (not bounding boxes)
- 💰 **Cost Estimation Engine** — Converts mask pixel area → severity level → repair cost range (USD)
- 🌐 **Streamlit Web App** — Upload any vehicle image and get results in seconds at `localhost:8501`
- 🖥️ **CLI Support** — Terminal-based analysis with `--input` and `--model` flags via argparse
- 🧪 **Data Augmentation Pipeline** — Horizontal flips, rotations, and HSV brightness variations for robust training
- 📊 **Metrics Tracking** — mAP and IoU tracked automatically during YOLOv8 training

---

## 🖼️ Supported Parts & Cost Matrix

The estimator currently supports three vehicle parts across three severity levels:

| Part | Minor Scratch | Moderate Dent | Severe Structural |
|------|:---:|:---:|:---:|
| **Bumper** | $150 – $300 | $350 – $500 | $600 – $1,000 |
| **Windshield** | $50 – $150 | $200 – $300 | $300 – $600 |
| **Side Door** | $200 – $400 | $450 – $700 | $800 – $1,500 |

### Severity Classification (based on mask pixel area at 640×640 resolution)

| Severity | Pixel Area | Description |
|----------|:---:|---|
| 🟢 Minor Scratch | < 5,000 px | Light surface scuff or scratch |
| 🟡 Moderate Dent | 5,000 – 25,000 px | Dented panel, cracked glass |
| 🔴 Severe Structural | > 25,000 px | Collapsed frame, full replacement needed |

---

## 🛠️ Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.10+ | Core language |
| Ultralytics YOLOv8-seg | 8.1.29 | Instance segmentation model |
| OpenCV | 4.9.0.80 | Image pre-processing & augmentation |
| NumPy | 1.24.3 | Array operations & normalization |
| Matplotlib | 3.8.3 | Visualization |
| Streamlit | Latest | Web UI |

---

## 🗂️ File Structure

```
Vehicle-Damage-Estimator/
│
├── app.py                  # Streamlit web interface
├── main.py                 # CLI entry point (argparse)
├── cost_estimator.py       # DamageEstimator class & cost matrix logic
├── preprocess.py           # Image normalization & augmentation pipeline
├── train.py                # YOLOv8-seg model training script
│
├── requirements.txt        # Python dependencies
├── Project_Report.md       # Detailed project report
│
├── test_car.webp           # Sample test image
└── Car-accident-uy_e.webp  # Demo vehicle image
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tejashri-del098/Vehicle-Damage-Estimator.git
cd Vehicle-Damage-Estimator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
opencv-python==4.9.0.80
numpy==1.24.3
ultralytics==8.1.29
matplotlib==3.8.3
streamlit
```

---

## 🚀 Usage

### Option A — Streamlit Web App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a damaged vehicle image (`.jpg`, `.jpeg`, `.png`, `.webp`) and the app will display:
- **Detected Part** — the identified vehicle component
- **Damage Severity** — Minor Scratch / Moderate Dent / Severe Structural Damage
- **Estimated Repair Cost** — USD range based on industry averages

### Option B — Command Line Interface

```bash
python main.py --input path/to/car_image.jpg --model path/to/yolov8_model.pt
```

**Arguments:**

| Flag | Required | Description |
|------|:---:|---|
| `--input` | ✅ | Path to the input vehicle image |
| `--model` | ✅ | Path to the trained YOLOv8-seg `.pt` weights file |

**Example output:**
```
--- Vehicle Damage Estimator ---
Input Image: car_image.jpg
Model File:  yolov8_model.pt

Running inference...

--- ESTIMATE REPORT ---
Detected Part: Bumper
Damage Severity: Moderate Dent
Estimated Repair Cost: $350 - $500
-----------------------

[Technical Info: Detected Part Area refers to 15000 pixels via Instance Segmentation MASK]
```

### Option C — Test the Estimator Directly

```bash
python cost_estimator.py
```

Runs a standalone test of the estimation logic (Bumper, 12,000 px → Moderate Dent → $350–$500).

---

## 🔬 How It Works — Pipeline

### Phase 1: Data Pre-processing (`preprocess.py`)

All input images are standardized to **640×640** arrays with pixel values normalized from `[0, 255]` to `[0.0, 1.0]`. A data augmentation suite is applied during training to improve generalization:

| Augmentation | Details |
|---|---|
| Horizontal Flip | Mirror image left-right |
| Rotation | −15° planar rotation |
| Brightness (bright) | HSV V-channel +50 |
| Brightness (dark) | HSV V-channel −50 |

```python
# preprocess.py — augment_image() produces 5 variants per image:
# base, flipped, bright, dark, rotated
augmented_samples = augment_image(resized_img)
```

### Phase 2: Model Training (`train.py`)

Uses **YOLOv8n-seg** (nano segmentation) as the base pretrained model, fine-tuned on vehicle damage data:

```bash
python train.py --data data.yaml --epochs 50
```

Training is configured via a `data.yaml` file specifying:
- **Classes:** `[Bumper, Windshield, Side Door]`
- **Image size:** 640×640
- **Metrics:** mAP (Mean Average Precision) + IoU (Intersection over Union) — tracked automatically

Results are saved to `Damage_Estimator/segmentation_v1/`.

**Why Instance Segmentation over Object Detection?**
> Bounding boxes include background pixels — useless for damage area measurement. YOLOv8-seg produces pixel-accurate masks, so the pixel count directly represents the physical surface area of damage. More pixels = larger damage = higher repair cost.

### Phase 3: Cost Estimation (`cost_estimator.py`)

The `DamageEstimator` class exposes two methods:

```python
estimator = DamageEstimator()

# Step 1: classify severity from mask pixel area
severity = estimator.estimate_severity(pixel_area=15000)
# → "Moderate Dent"

# Step 2: look up cost range from static cost matrix
report = estimator.get_estimate(part_name="Bumper", mask_area_pixels=15000)
# → { "Detected Part": "Bumper", "Damage Severity": "Moderate Dent", "Estimated Repair Cost": "$350 - $500" }
```

### Phase 4: Web UI / CLI Output (`app.py` / `main.py`)

- **Streamlit** (`app.py`) — renders results as interactive metric cards in the browser
- **CLI** (`main.py`) — prints a formatted estimate report to the terminal using argparse

---

## 📈 Model Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mAP** (Mean Average Precision) | Overall detection accuracy across all part classes |
| **IoU** (Intersection over Union) | Accuracy of mask overlap with ground truth damage region |

Both are tracked automatically by Ultralytics during training validation.

---

## 🔮 Future Improvements

- [x] Integrate live YOLOv8 inference (replace demo mock with real model weights)
- [ ] Expand cost matrix to cover more vehicle parts (Hood, Fender, Roof, Headlights)
- [ ] Add confidence score display from YOLO predictions
- [ ] Support multi-damage detection in a single image
- [ ] Add Indian repair cost estimates (INR) alongside USD
- [ ] Export PDF/CSV damage report

---

## 👩‍💻 Author

**Tejashri** — [GitHub @tejashri-del098](https://github.com/tejashri-del098)

---

## 📄 License

This project is open source. See the repository for license details.
