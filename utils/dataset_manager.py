import os
import shutil
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

class DatasetManager:
    def __init__(self):
        self.yolo_train_path = Path("data/images/train")
        self.yolo_val_path = Path("data/images/val")
        self.yolo_train_labels = Path("data/labels/train")
        self.yolo_val_labels = Path("data/labels/val")
        self.classification_path = Path("data/coral_types")
        self.coral_types = ["brain_coral", "staghorn_coral", "table_coral", "soft_coral", "fan_coral", "mushroom_coral"]
        
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.yolo_train_path,
            self.yolo_val_path,
            self.yolo_train_labels,
            self.yolo_val_labels,
            self.classification_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create coral type directories
        for coral_type in self.coral_types:
            (self.classification_path / coral_type).mkdir(exist_ok=True)
    
    def get_dataset_stats(self):
        """Get current dataset statistics"""
        stats = {
            "yolo": {
                "train_images": len(list(self.yolo_train_path.glob("*.jpg"))),
                "train_labels": len(list(self.yolo_train_labels.glob("*.txt"))),
                "val_images": len(list(self.yolo_val_path.glob("*.jpg"))),
                "val_labels": len(list(self.yolo_val_labels.glob("*.txt")))
            },
            "classification": {}
        }
        
        for coral_type in self.coral_types:
            type_path = self.classification_path / coral_type
            if type_path.exists():
                stats["classification"][coral_type] = len(list(type_path.glob("*.jpg")))
            else:
                stats["classification"][coral_type] = 0
        
        return stats
    
    def add_yolo_image(self, uploaded_file, annotation_text, split="train"):
        """Add a new image to YOLO dataset with annotation"""
        try:
            # Determine paths
            if split == "train":
                img_path = self.yolo_train_path
                label_path = self.yolo_train_labels
            else:
                img_path = self.yolo_val_path
                label_path = self.yolo_val_labels
            
            # Generate filename
            existing_files = len(list(img_path.glob("*.jpg")))
            filename = f"coral_{split}_{existing_files:03d}"
            
            # Save image
            image = Image.open(uploaded_file)
            # Resize to standard YOLO size
            image = image.resize((640, 640))
            img_file_path = img_path / f"{filename}.jpg"
            image.save(img_file_path)
            
            # Save annotation
            label_file_path = label_path / f"{filename}.txt"
            with open(label_file_path, "w") as f:
                f.write(annotation_text)
            
            return True, f"Added {filename}.jpg to {split} dataset"
            
        except Exception as e:
            return False, f"Error adding image: {str(e)}"
    
    def add_classification_image(self, uploaded_file, coral_type):
        """Add a new image to classification dataset"""
        try:
            if coral_type not in self.coral_types:
                return False, f"Invalid coral type: {coral_type}"
            
            type_path = self.classification_path / coral_type
            existing_files = len(list(type_path.glob("*.jpg")))
            filename = f"{coral_type}_{existing_files:03d}.jpg"
            
            # Save image
            image = Image.open(uploaded_file)
            # Resize for classification
            image = image.resize((224, 224))
            img_file_path = type_path / filename
            image.save(img_file_path)
            
            return True, f"Added {filename} to {coral_type} dataset"
            
        except Exception as e:
            return False, f"Error adding classification image: {str(e)}"
    
    def validate_yolo_annotation(self, annotation_text):
        """Validate YOLO annotation format"""
        try:
            lines = annotation_text.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        return False, f"Invalid format: {line}. Expected: class x y w h"
                    
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    
                    if class_id != 0:
                        return False, f"Invalid class ID: {class_id}. Use 0 for coral"
                    
                    if not all(0 <= val <= 1 for val in [x, y, w, h]):
                        return False, f"Coordinates must be between 0 and 1: {line}"
            
            return True, "Annotation format is valid"
            
        except Exception as e:
            return False, f"Annotation validation error: {str(e)}"
    
    def update_dataset_info(self):
        """Update dataset info JSON file"""
        stats = self.get_dataset_stats()
        
        dataset_info = {
            "last_updated": str(Path().resolve()),
            "yolo_detection": {
                "total_train_images": stats["yolo"]["train_images"],
                "total_val_images": stats["yolo"]["val_images"],
                "annotation_format": "YOLO (normalized coordinates)",
                "classes": ["coral"]
            },
            "classification": {
                "total_images": sum(stats["classification"].values()),
                "coral_types": self.coral_types,
                "distribution": stats["classification"]
            }
        }
        
        with open("data/dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info
    
    def create_annotation_example(self):
        """Create example annotation text for users"""
        example = """# YOLO Annotation Format (one line per coral detection)
# Format: class_id center_x center_y width height
# All values are normalized (0.0 to 1.0)
# class_id: 0 (coral)
# center_x, center_y: center point of bounding box
# width, height: box dimensions

# Example annotations:
0 0.5 0.3 0.2 0.4
0 0.7 0.6 0.15 0.25"""
        return example