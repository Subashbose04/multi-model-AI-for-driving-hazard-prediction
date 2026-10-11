import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


class HazardPredictionModel(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        
        self.yolo = YOLO('yolov8n.pt') 
        
        
        model_id = "Salesforce/blip-image-captioning-base"
        self.blip_processor = BlipProcessor.from_pretrained(model_id)
        self.blip = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.blip.eval()

    @torch.no_grad()
    def get_bounding_boxes(self, image_pil, max_boxes=15):
        """
        Runs YOLOv8 and returns a list of crop PIL images and boxes.
        We return actual box coordinates and labels for drawing.
        """
        image_np = np.array(image_pil.convert("RGB"))
        results = self.yolo(image_np, verbose=False)
        
        boxes_info = []
        if len(results) > 0:
            result = results[0]
            bboxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            names = result.names
            
           
            if len(bboxes) > 0:
                indices = np.argsort(confs)[::-1][:max_boxes]
                for idx in indices:
                    x1, y1, x2, y2 = map(int, bboxes[idx])
                    conf = float(confs[idx])
                    label = names[int(cls_ids[idx])]
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        boxes_info.append({
                            "coords": [x1, y1, x2, y2],
                            "label": label,
                            "confidence": conf
                        })
        return boxes_info
        
    @torch.no_grad()
    def generate(self, image_pil, max_new_tokens=50):
        """
        End-to-end hazard prediction generation from a single image.
        Uses BLIP directly for scene understanding.
        """
        
        boxes_info = self.get_bounding_boxes(image_pil)
        
        
        text = "A photo of a view from a driving car:"
        inputs = self.blip_processor(images=image_pil, text=text, return_tensors="pt").to(self.device)
        
        outputs = self.blip.generate(
            **inputs, 
            max_new_tokens=40,
            repetition_penalty=1.5
        )
        
        scene_context = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        if scene_context.lower().startswith(text.lower().strip()):
            scene_context = scene_context[len(text.strip()):].strip()
            
        
        risk_level = "Low"
        probability = 5
        hazards = []
        
        for box in boxes_info:
            label = box["label"].lower()
            conf = box["confidence"]
            
            
            if label in ['person', 'bicycle', 'motorcycle']:
                hazards.append(f"pedestrian/cyclist ({conf*100:.0f}%)")
                risk_level = "High"
                probability = max(probability, int(min(95, conf * 100 + 35))) # Increased sensitivity
            elif label in ['car', 'truck', 'bus']:
                hazards.append(f"vehicle ({conf*100:.0f}%)")
                if risk_level != "High":
                    risk_level = "Moderate"
                probability = max(probability, int(min(85, conf * 100 + 25))) # Increased sensitivity
            elif label in ['stop sign', 'traffic light']:
                hazards.append(f"traffic control ({conf*100:.0f}%)")
                
        if len(hazards) > 0:
            hazard_str = ", ".join(hazards)
            generated_text = (
                f"**Risk Level:** {risk_level} (Collision Probability: {probability}%)\n\n"
                f"**Detected Hazards:** {hazard_str}\n\n"
                f"**Scene Context:** {scene_context.capitalize()}"
            )
        else:
            generated_text = (
                f"**Risk Level:** Low (Collision Probability: <5%)\n\n"
                f"**Detected Hazards:** Clear path ahead\n\n"
                f"**Scene Context:** {scene_context.capitalize()}"
            )
            
        return generated_text, boxes_info

    def analyze_video_frames(self, frames, progress_callback=None):
        """
        Takes a list of PIL Images (extracted from video).
        Runs analysis on each and returns results list.
        """
        results = []
        for i, frame in enumerate(frames):
            caption, boxes = self.generate(frame)
            results.append({
                "frame_idx": i,
                "caption": caption,
                "boxes": boxes
            })
            if progress_callback:
                progress_callback(i + 1, len(frames))
        return results
