import argparse
import torch
from PIL import Image, ImageDraw
from model import HazardPredictionModel
import cv2
import numpy as np

def predict(image_path, device='cpu'):
    print(f"Loading image from {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Initializing HazardPredictionModel...")
    model = HazardPredictionModel(device=device)
    
    print("Generating prediction...")
    text, bboxes = model.generate(image)
    
    print(f"\n--- Prediction Result ---")
    print(f"Predicted Hazard: {text}")
    print(f"Detected Visual Entities (Boxes): {len(bboxes)}")
    
    # Draw boxes for visualization
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(bboxes):
        if i >= 3: 
            break
        # x1, y1, x2, y2
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"Entity {i+1}", fill="white")
        
    out_path = "output_prediction.jpg"
    image.save(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hazard Prediction on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    
    predict(args.image, args.device)
