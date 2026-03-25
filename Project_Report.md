# Project Report: Vehicle Damage Estimator

## 1. Phase 1: Data Acquisition & Pre-processing (Vision Foundation)
To build a robust foundation, we implemented a data pre-processing pipeline designed to construct a clean dataset containing vehicle images with varying scenarios of damage. I incorporated **Feature Extraction** and **Image Filtering** via OpenCV algorithms. 

- **Normalization and Resizing**: All inputs are standardized to 640x640 arrays with normalized floating-point intervals (0.0 to 1.0). This significantly improves the convergence capabilities of deep neural network optimizers.
- **Data Augmentation**: A robust series of filters altering horizontal flips, planar rotations (-15 degrees), and HSV value variations (brightness extraction and enhancement) were created. This is crucial for avoiding overfitting, and allowing the model to recognize damage irrespective of harsh sunlight, glares, or misaligned photos.

## 2. Phase 2: Model Training & Instance Segmentation (AI/ML Core)
To handle damage detection precisely, I migrated away from standard Object Detection (Bounding Boxes) toward **Instance Segmentation** using Ultralytics YOLOv8-seg. 

- **Model Selection & Utility**: YOLO-based segmentation models not only detect regions of interest but output pixel-dense masks corresponding exclusively to the damaged structure (e.g., specific area of a bumper or side door). This granularity is paramount since surface area inherently correlates with repair hours.
- **Metrics Tracking**: Progress throughout multi-class evaluation was formulated by tracking **mAP (Mean Average Precision)** and **Intersection over Union (IoU)**, validating precision boundaries between true positives and background noise.

## 3. Phase 3: CLI Development & Estimation Logic
Building the final interface involved assembling a terminal-based CLI utilizing Python's `argparse` library. This ensured seamless access using straightforward command syntax.

- **Estimation Engine (`cost_estimator.py`)**: The underlying function computes a heuristic evaluating the mask's surface pixel area to categorize severity levels (Minor Scratch vs Moderate Dent). It then dynamically maps this severity alongside the identified physical structure (e.g., 'Bumper') to output an empirical average cost estimation range.
- **Formatted Outputs**: The final terminal reports summarize strictly graded properties formatted as *Detected Part*, *Damage Severity*, and *Estimated Repair Cost*.

## 4. Reflection
This project exposed me to the entire lifecycle of a computer vision project, blending raw signal processing techniques with modern Deep Learning backbones. Relying initially on OpenCV procedures underscored how basic filtering shapes model efficiency. Developing the mask-to-cost heuristic explicitly demonstrated how AI parameters like 'pixel masks' are converted into tangible, real-world utility frameworks. I also appreciated how architecting standard command line interfaces ensures the scalability and usability of algorithmic prototypes beyond standard theoretical settings. All code is structured properly to enforce clarity and reusability. 
