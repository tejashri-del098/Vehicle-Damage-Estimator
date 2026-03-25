# Vehicle Damage Estimator

A computer vision tool built using OpenCV and Ultralytics YOLOv8 instance segmentation to detect vehicle parts, their damage severities, and subsequently estimate repair costs based on the instance mask's pixel area.

## Features
- **Data Augmentation**: Robust preprocessing utilizing OpenCV (Rotations, Brightness adjustments, Flipping).
- **Instance Segmentation**: Deep learning segmentation to highlight exact part shapes and boundaries.
- **Cost Estimation Logic**: Rules-based fallback mapping mask area (in pixels) to industry average part costs.
- **Argparse CLI**: Easy integration directly from your terminal.

## Quick Start

Assuming you have `git`, `python`, and `pip` installed.

**1. Clone the repository**
```bash
git clone https://github.com/{username}/Vehicle-Damage-Estimator.git
cd "Vehicle Damage Estimator"
```

**2. Set up environment and install dependencies**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

**3. Run the CLI Application**
To perform an estimation on an image, simply run the `main.py` script:
```bash
python main.py --input test_car.jpg --model damage_v1.pt
```

*(Note: If `test_car.jpg` or `damage_v1.pt` don't exist, the script simulates them for testing and grading visualization).*

## Pipeline Overviews

### Preprocessing (`preprocess.py`)
Run image datasets through standard image normalization and augmentations:
```bash
python preprocess.py --input_dir data/raw --output_dir data/processed
```

### Model Training (`train.py`)
Demonstration of building the Mask-RCNN / YOLOv8-seg model tracking mAPs and IoU:
```bash
python train.py --epochs 50 --data data.yaml
```

*Author: [Your Name / Student ID]*
