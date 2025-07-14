#!/usr/bin/env python3
"""
Create a simple demo coral detection system for testing the interface
"""

import cv2
import numpy as np
import json
from pathlib import Path

def create_simple_coral_detector():
    """Create a simple coral detection that finds coral-colored regions"""
    
    def detect_coral_regions(image):
        """Simple coral detection based on color"""
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define coral color ranges (HSV)
        coral_ranges = [
            # Orange/coral colors
            ([0, 50, 50], [25, 255, 255]),
            # Red colors  
            ([160, 50, 50], [180, 255, 255]),
            # Yellow-green (some corals)
            ([25, 50, 50], [85, 255, 255])
        ]
        
        for i, (lower, upper) in enumerate(coral_ranges):
            # Create mask for this color range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum coral size
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on area and color concentration
                    roi_mask = mask[y:y+h, x:x+w]
                    color_ratio = np.sum(roi_mask > 0) / (w * h)
                    confidence = min(0.95, 0.3 + color_ratio * 0.6)
                    
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': confidence
                    })
        
        return detections
    
    return detect_coral_regions

def create_demo_yolo_model():
    """Create a demo YOLO model file"""
    model_info = {
        "model_type": "YOLOv8_demo",
        "trained_on": "synthetic_coral_dataset", 
        "classes": ["coral"],
        "note": "This is a demo model for interface testing"
    }
    
    # Save model info
    with open("trained_models/demo_yolo_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("✓ Demo YOLO model info created")

def main():
    """Create demo detection system"""
    print("🎯 Creating Demo Coral Detection System")
    print("=" * 50)
    
    # Ensure trained_models directory exists
    Path("trained_models").mkdir(exist_ok=True)
    
    # Create demo YOLO model info
    create_demo_yolo_model()
    
    print("✓ Demo detection system ready!")
    print("This provides basic coral detection for testing the interface.")

if __name__ == "__main__":
    main()