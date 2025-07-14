#!/usr/bin/env python3
"""
Generate sample coral reef training data for the coral detection and classification system.
This script creates synthetic coral reef images with proper annotations for YOLO training
and organizes coral type images for classification training.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter
import math

class CoralDataGenerator:
    def __init__(self):
        self.output_base = Path("data")
        self.image_size = (640, 640)
        # BGR color format for OpenCV
        self.coral_colors = [
            (84, 106, 255),   # Coral red (BGR)
            (122, 160, 255),  # Light salmon (BGR)
            (185, 218, 255),  # Peach puff (BGR)
            (181, 228, 255),  # Moccasin (BGR)
            (213, 239, 255),  # Papaya whip (BGR)
            (140, 230, 240),  # Khaki (BGR)
            (144, 238, 144),  # Light green (BGR)
            (230, 216, 173),  # Light blue (BGR)
            (221, 160, 221),  # Plum (BGR)
            (203, 192, 255),  # Pink (BGR)
        ]
        
        self.coral_types = [
            "brain_coral",
            "staghorn_coral", 
            "table_coral",
            "soft_coral",
            "fan_coral",
            "mushroom_coral"
        ]
        
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        # YOLO training directories
        (self.output_base / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_base / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_base / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_base / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Classification directories
        for coral_type in self.coral_types:
            (self.output_base / "coral_types" / coral_type).mkdir(parents=True, exist_ok=True)
    
    def generate_underwater_background(self, width, height):
        """Generate realistic underwater background"""
        # Create blue-green underwater gradient
        background = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Blue-green gradient
        for y in range(height):
            intensity = int(20 + (y / height) * 40)  # Darker at bottom
            blue_val = min(255, 60 + intensity)
            green_val = min(255, 80 + intensity // 2)
            red_val = min(255, 20 + intensity // 4)
            background[y, :] = [red_val, green_val, blue_val]
        
        # Add some texture/noise
        noise = np.random.randint(-20, 20, (height, width, 3))
        background = np.clip(background.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some light rays
        for _ in range(3):
            x = random.randint(0, width-1)
            y = random.randint(0, height // 2)
            radius = random.randint(30, 80)
            # BGR color format for OpenCV
            color = (int(min(255, background[y, x, 0] + 20)),  # Blue
                    int(min(255, background[y, x, 1] + 30)),   # Green  
                    int(min(255, background[y, x, 2] + 30)))   # Red
            cv2.circle(background, (x, y), radius, color, -1)
        
        # Blur the light rays
        background = cv2.GaussianBlur(background, (15, 15), 0)
        
        return background
    
    def generate_coral_shape(self, coral_type, size):
        """Generate different coral shapes based on type"""
        mask = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        
        if coral_type == "brain_coral":
            # Brain coral: round with wavy texture
            cv2.circle(mask, (center, center), center - 5, 255, -1)
            # Add wavy patterns
            for i in range(5):
                angle = i * 72
                x = int(center + (center - 15) * math.cos(math.radians(angle)))
                y = int(center + (center - 15) * math.sin(math.radians(angle)))
                cv2.circle(mask, (x, y), 8, 0, -1)
        
        elif coral_type == "staghorn_coral":
            # Staghorn: branching structure
            cv2.circle(mask, (center, center), 15, 255, -1)
            for i in range(6):
                angle = i * 60 + random.randint(-15, 15)
                length = random.randint(20, center - 10)
                end_x = int(center + length * math.cos(math.radians(angle)))
                end_y = int(center + length * math.sin(math.radians(angle)))
                cv2.line(mask, (center, center), (end_x, end_y), 255, 8)
                cv2.circle(mask, (end_x, end_y), 5, 255, -1)
        
        elif coral_type == "table_coral":
            # Table coral: flat, wide structure
            cv2.ellipse(mask, (center, center), (center - 5, center // 2), 0, 0, 360, 255, -1)
        
        elif coral_type == "soft_coral":
            # Soft coral: irregular, flowing shape
            points = []
            for i in range(8):
                angle = i * 45
                radius = random.randint(center // 2, center - 5)
                x = int(center + radius * math.cos(math.radians(angle)))
                y = int(center + radius * math.sin(math.radians(angle)))
                points.append([x, y])
            cv2.fillPoly(mask, [np.array(points)], 255)
        
        elif coral_type == "fan_coral":
            # Fan coral: fan-like structure
            cv2.ellipse(mask, (center, center), (center - 5, center - 15), 0, 0, 180, 255, -1)
            for i in range(5):
                angle = -90 + i * 45
                end_x = int(center + (center - 5) * math.cos(math.radians(angle)))
                end_y = int(center + (center - 5) * math.sin(math.radians(angle)))
                cv2.line(mask, (center, center), (end_x, end_y), 255, 3)
        
        else:  # mushroom_coral
            # Mushroom coral: dome shape
            cv2.ellipse(mask, (center, center), (center - 5, center // 3), 0, 0, 360, 255, -1)
            cv2.circle(mask, (center, center - center // 4), center - 10, 255, -1)
        
        return mask
    
    def generate_coral_image(self, coral_type, color, size):
        """Generate a single coral image with given type and color"""
        # Create coral shape mask
        mask = self.generate_coral_shape(coral_type, size)
        
        # Create colored coral
        coral_img = np.zeros((size, size, 3), dtype=np.uint8)
        coral_img[mask > 0] = color
        
        # Add some texture
        texture = np.random.randint(-20, 20, (size, size, 3))
        coral_img = np.clip(coral_img.astype(int) + texture, 0, 255).astype(np.uint8)
        coral_img[mask == 0] = [0, 0, 0]  # Keep background transparent
        
        # Add some highlights and shadows
        highlight_mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1) - mask
        coral_img[highlight_mask > 0] = np.clip(coral_img[highlight_mask > 0] + 30, 0, 255)
        
        return coral_img, mask
    
    def place_coral_on_background(self, background, coral_img, mask, x, y):
        """Place coral image on background at specified position"""
        h, w = coral_img.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Ensure coral fits within background
        if x + w > bg_w or y + h > bg_h or x < 0 or y < 0:
            return background, None
        
        # Blend coral with background
        coral_region = background[y:y+h, x:x+w].copy()
        coral_region[mask > 0] = coral_img[mask > 0]
        background[y:y+h, x:x+w] = coral_region
        
        # Return bounding box in YOLO format (normalized)
        center_x = (x + w // 2) / bg_w
        center_y = (y + h // 2) / bg_h
        width = w / bg_w
        height = h / bg_h
        
        return background, (center_x, center_y, width, height)
    
    def generate_yolo_training_image(self, img_id):
        """Generate a complete training image with multiple corals and annotations"""
        # Create underwater background
        background = self.generate_underwater_background(*self.image_size)
        
        annotations = []
        coral_instances = []
        
        # Add 2-5 corals per image
        num_corals = random.randint(2, 5)
        
        for _ in range(num_corals):
            # Random coral properties
            coral_type = random.choice(self.coral_types)
            coral_color = random.choice(self.coral_colors)
            coral_size = random.randint(60, 120)
            
            # Generate coral
            coral_img, mask = self.generate_coral_image(coral_type, coral_color, coral_size)
            
            # Random position (ensure it fits)
            max_x = self.image_size[0] - coral_size
            max_y = self.image_size[1] - coral_size
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Place coral and get bounding box
            background, bbox = self.place_coral_on_background(background, coral_img, mask, x, y)
            
            if bbox:
                # YOLO annotation format: class_id center_x center_y width height
                annotations.append(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                coral_instances.append({
                    'type': coral_type,
                    'bbox': (x, y, coral_size, coral_size),
                    'color': coral_color
                })
        
        return background, annotations, coral_instances
    
    def generate_classification_image(self, coral_type, img_id):
        """Generate a single coral image for classification training"""
        size = random.randint(150, 300)
        color = random.choice(self.coral_colors)
        
        # Generate coral
        coral_img, mask = self.generate_coral_image(coral_type, color, size)
        
        # Create underwater background for just this coral
        bg_size = size + 20
        background = self.generate_underwater_background(bg_size, bg_size)
        
        # Place coral in center
        x, y = 10, 10
        final_img, _ = self.place_coral_on_background(background, coral_img, mask, x, y)
        
        return final_img
    
    def generate_dataset(self, num_yolo_images=40, num_classification_per_type=15):
        """Generate complete dataset for both YOLO and classification"""
        print(f"Generating {num_yolo_images} YOLO training images...")
        
        # Generate YOLO training data
        for i in range(num_yolo_images):
            img, annotations, coral_instances = self.generate_yolo_training_image(i)
            
            # Determine if this goes to train or val (80/20 split)
            if i < int(num_yolo_images * 0.8):
                subset = "train"
            else:
                subset = "val"
            
            # Save image
            img_path = self.output_base / "images" / subset / f"coral_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Save annotations
            if annotations:
                label_path = self.output_base / "labels" / subset / f"coral_{i:03d}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_yolo_images} YOLO images")
        
        print(f"\nGenerating {num_classification_per_type} images per coral type...")
        
        # Generate classification training data
        for coral_type in self.coral_types:
            for i in range(num_classification_per_type):
                img = self.generate_classification_image(coral_type, i)
                
                # Save image
                img_path = self.output_base / "coral_types" / coral_type / f"{coral_type}_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
            
            print(f"  Generated {num_classification_per_type} images for {coral_type}")
        
        print("\n✓ Dataset generation complete!")
        print(f"  YOLO: {num_yolo_images} annotated images")
        print(f"  Classification: {len(self.coral_types)} types × {num_classification_per_type} images")

def main():
    """Generate sample coral reef dataset"""
    print("🐠 Generating Coral Reef Training Dataset")
    print("=" * 50)
    
    generator = CoralDataGenerator()
    generator.generate_dataset(
        num_yolo_images=40,  # More than 30 as requested
        num_classification_per_type=15  # Good variety for each coral type
    )
    
    print("\n🎯 Ready to train coral detection and classification models!")

if __name__ == "__main__":
    main()