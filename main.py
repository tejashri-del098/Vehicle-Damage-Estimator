import argparse
import os
import sys
from cost_estimator import DamageEstimator

from ultralytics import YOLO

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Simulating testing image at {image_path}. (In real life, provide a valid image).")
        with open(image_path, "wb") as f:
            f.write(b"Mock image data")
    return True

def analyze_damage(image_path, model_path):
    print(f"\n--- Vehicle Damage Estimator ---")
    print(f"Input Image: {image_path}")
    
    # Fallback to pretrained model if dummy is provided
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        print(f"Model File {model_path} is invalid/dummy. Auto-loading 'yolov8n-seg.pt'...")
        model_path = "yolov8n-seg.pt"
    else:
        print(f"Model File:  {model_path}\n")
        
    estimator = DamageEstimator()
    print("Loading model and running real YOLOv8 inference...")
    
    try:
        model = YOLO(model_path)
        results = model(image_path)
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return
        
    predictions_found = False
    result = results[0]
    
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data):
            predictions_found = True
            pixel_area = int(mask.sum().item())
            cls = int(result.boxes.cls[i].item())
            part_name = model.names[cls]
            
            report = estimator.get_estimate(part_name, pixel_area)
            if report:
                print("\n--- ESTIMATE REPORT ---")
                for k, v in report.items():
                    print(f"{k}: {v}")
                print("-----------------------\n")
                print(f"[Technical Info: Detected '{part_name}' Area spans {pixel_area} pixels]")
                
    if not predictions_found:
        print("Model did not detect any objects or damage masks in this image.")

def main():
    parser = argparse.ArgumentParser(description="Estimate Vehicle Damage Repair Cost from an Image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input car image.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLOv8 model (.pt).")
    
    args = parser.parse_args()
    
    # Ensure dummy image exists if specified
    load_image(args.input)
    
    analyze_damage(args.input, args.model)

if __name__ == "__main__":
    main()
