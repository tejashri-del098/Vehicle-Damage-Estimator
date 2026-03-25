import cv2
import numpy as np
import os
import argparse
import glob

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Reads an image, resizes it, and normalizes the pixel values.
    This step covers basic Feature Extraction and Image Filtering setup
    for Deep Learning models by creating a standardized input.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Resize image to target dimensions
    resized = cv2.resize(image, target_size)

    # Normalize pixel values (0-255 -> 0.0-1.0)
    normalized = resized.astype('float32') / 255.0

    return resized, normalized

def augment_image(image):
    """
    Applies Data Augmentation: rotation, flipping, brightness adjustment.
    Helps the model recognize damage in various weather/lighting conditions.
    """
    augmented_images = []

    # 1. Base Image
    augmented_images.append(("base", image))

    # 2. Horizontal Flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(("flipped", flipped))

    # 3. Brightness Adjustment (Brighter)
    # Convert to HSV to safely adjust brightness (V channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Add brightness, capping at 255
    v_bright = cv2.add(v, 50)
    final_hsv_bright = cv2.merge((h, s, v_bright))
    bright_img = cv2.cvtColor(final_hsv_bright, cv2.COLOR_HSV2BGR)
    augmented_images.append(("bright", bright_img))

    # 4. Brightness Adjustment (Darker)
    v_dark = cv2.subtract(v, 50)
    final_hsv_dark = cv2.merge((h, s, v_dark))
    dark_img = cv2.cvtColor(final_hsv_dark, cv2.COLOR_HSV2BGR)
    augmented_images.append(("dark", dark_img))

    # 5. Rotation (-15 degrees)
    rows, cols = image.shape[:2]
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    rotated = cv2.warpAffine(image, M_rot, (cols, rows))
    augmented_images.append(("rotated", rotated))

    return augmented_images

def main():
    parser = argparse.ArgumentParser(description="Vehicle Damage Preprocessing Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output augmented images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(args.input_dir, "*.[jp][pn]*[g]")) # matches jpg, png, jpeg

    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        # 1. Feature Extraction & Normalization
        resized_img, norm_img = preprocess_image(img_path)
        if resized_img is None:
            continue

        # 2. Data Augmentation
        augmented_samples = augment_image(resized_img)

        # Save all augmented samples
        for aug_name, aug_img in augmented_samples:
            out_name = f"{name}_{aug_name}{ext}"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, aug_img)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
