import os
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
    def validate_image_directory(self, directory_path):
        """Validate image directory structure"""
        path = Path(directory_path)
        if not path.exists():
            return False, f"Directory {directory_path} does not exist"
        
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(path.glob(f"*{ext}"))
            image_files.extend(path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            return False, f"No image files found in {directory_path}"
        
        return True, f"Found {len(image_files)} images"
    
    def resize_images(self, input_dir, output_dir, target_size=(640, 640)):
        """Resize images to target size for YOLO training"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        error_count = 0
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in self.supported_formats:
                try:
                    # Read image
                    img = cv2.imread(str(image_file))
                    if img is None:
                        print(f"Failed to read {image_file}")
                        error_count += 1
                        continue
                    
                    # Resize image
                    resized_img = cv2.resize(img, target_size)
                    
                    # Save resized image
                    output_file = output_path / image_file.name
                    cv2.imwrite(str(output_file), resized_img)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    error_count += 1
        
        return processed_count, error_count
    
    def augment_images(self, input_dir, output_dir, augment_factor=2):
        """Apply data augmentation to increase dataset size"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy original images first
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in self.supported_formats:
                shutil.copy2(image_file, output_path / image_file.name)
        
        # Apply augmentations
        augmented_count = 0
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in self.supported_formats:
                try:
                    img = cv2.imread(str(image_file))
                    if img is None:
                        continue
                    
                    base_name = image_file.stem
                    extension = image_file.suffix
                    
                    for i in range(augment_factor):
                        # Apply random augmentations
                        augmented_img = self.apply_augmentation(img)
                        
                        # Save augmented image
                        aug_filename = f"{base_name}_aug_{i}{extension}"
                        aug_path = output_path / aug_filename
                        cv2.imwrite(str(aug_path), augmented_img)
                        augmented_count += 1
                        
                except Exception as e:
                    print(f"Error augmenting {image_file}: {e}")
        
        return augmented_count
    
    def apply_augmentation(self, image):
        """Apply random augmentation to an image"""
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        return augmented
    
    def create_yolo_dataset_split(self, images_dir, labels_dir, train_ratio=0.8):
        """Create train/val split for YOLO dataset"""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        # Split files
        train_files, val_files = train_test_split(
            image_files, train_size=train_ratio, random_state=42
        )
        
        # Create directories
        train_img_dir = Path("data/images/train")
        val_img_dir = Path("data/images/val")
        train_label_dir = Path("data/labels/train")
        val_label_dir = Path("data/labels/val")
        
        for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Move files to appropriate directories
        self._move_files_with_labels(train_files, train_img_dir, labels_path, train_label_dir)
        self._move_files_with_labels(val_files, val_img_dir, labels_path, val_label_dir)
        
        return len(train_files), len(val_files)
    
    def _move_files_with_labels(self, file_list, img_dest_dir, labels_src_dir, label_dest_dir):
        """Move image files and corresponding label files"""
        for img_file in file_list:
            # Copy image
            shutil.copy2(img_file, img_dest_dir / img_file.name)
            
            # Copy corresponding label file if exists
            label_name = img_file.stem + ".txt"
            label_file = labels_src_dir / label_name
            if label_file.exists():
                shutil.copy2(label_file, label_dest_dir / label_name)
    
    def validate_yolo_annotations(self, labels_dir):
        """Validate YOLO annotation format"""
        labels_path = Path(labels_dir)
        valid_count = 0
        invalid_count = 0
        errors = []
        
        for label_file in labels_path.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"{label_file.name}:{line_num} - Expected 5 values, got {len(parts)}")
                        invalid_count += 1
                        continue
                    
                    # Validate values
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        
                        # Check if coordinates are normalized (0-1)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"{label_file.name}:{line_num} - Coordinates not normalized")
                            invalid_count += 1
                            continue
                        
                        valid_count += 1
                        
                    except ValueError:
                        errors.append(f"{label_file.name}:{line_num} - Invalid numeric values")
                        invalid_count += 1
                        
            except Exception as e:
                errors.append(f"{label_file.name} - Error reading file: {e}")
                invalid_count += 1
        
        return valid_count, invalid_count, errors
