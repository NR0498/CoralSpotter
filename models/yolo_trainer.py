import os
import yaml
import shutil
from pathlib import Path
import subprocess
import sys

# Try to import ultralytics, fallback to basic implementation if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO training will use fallback implementation.")

class YOLOTrainer:
    def __init__(self):
        self.model_path = "trained_models/coral_yolo.pt"
        self.data_yaml_path = "data/coral_dataset.yaml"
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("trained_models", exist_ok=True)
        os.makedirs("data/images/train", exist_ok=True)
        os.makedirs("data/images/val", exist_ok=True)
        os.makedirs("data/labels/train", exist_ok=True)
        os.makedirs("data/labels/val", exist_ok=True)
        
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        dataset_config = {
            'train': 'data/images/train',
            'val': 'data/images/val',
            'nc': 1,  # number of classes (coral)
            'names': ['coral']
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
    
    def validate_dataset(self):
        """Validate that dataset has sufficient images"""
        train_images = len([f for f in os.listdir("data/images/train") 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if train_images < 30:
            return False, f"Insufficient training images. Found {train_images}, need at least 30."
        
        # Check for corresponding labels
        train_labels = len([f for f in os.listdir("data/labels/train") 
                           if f.lower().endswith('.txt')])
        
        if train_labels == 0:
            return False, "No label files found. Please annotate your images first."
        
        return True, f"Dataset validated: {train_images} training images, {train_labels} labels"
    
    def prepare_validation_split(self):
        """Create validation split if not exists"""
        train_dir = Path("data/images/train")
        val_dir = Path("data/images/val")
        train_labels_dir = Path("data/labels/train")
        val_labels_dir = Path("data/labels/val")
        
        # If validation directory is empty, move 20% of training data
        if len(list(val_dir.glob("*"))) == 0:
            train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.png"))
            val_split = int(len(train_images) * 0.2)
            
            for img_path in train_images[:val_split]:
                # Move image
                shutil.move(str(img_path), str(val_dir / img_path.name))
                
                # Move corresponding label if exists
                label_name = img_path.stem + ".txt"
                label_path = train_labels_dir / label_name
                if label_path.exists():
                    shutil.move(str(label_path), str(val_labels_dir / label_name))
    
    def train(self):
        """Train YOLO model"""
        try:
            # Validate dataset
            is_valid, message = self.validate_dataset()
            if not is_valid:
                return False, message
            
            # Create dataset configuration
            self.create_dataset_yaml()
            
            # Prepare validation split
            self.prepare_validation_split()
            
            if YOLO_AVAILABLE:
                # Load pre-trained YOLOv8 model
                model = YOLO('yolov8n.pt')  # Use nano version for faster training
                
                # Train the model
                results = model.train(
                    data=self.data_yaml_path,
                    epochs=50,
                    imgsz=640,
                    batch=8,
                    device='cpu',  # Use CPU for compatibility
                    project='runs/detect',
                    name='coral_detection',
                    save_period=10,
                    patience=10,
                    verbose=True
                )
                
                # Save the best model
                best_model_path = results.save_dir / 'weights' / 'best.pt'
                if os.path.exists(best_model_path):
                    shutil.copy(best_model_path, self.model_path)
                    return True, f"Model trained successfully and saved to {self.model_path}"
                else:
                    return False, "Training completed but best model not found"
            else:
                # Alternative training method using command line
                return self._train_with_pip_install()
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def _train_with_pip_install(self):
        """Alternative training method by installing ultralytics via pip"""
        try:
            # Try to install ultralytics via pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            
            # Re-import after installation
            from ultralytics import YOLO
            
            # Load pre-trained YOLOv8 model
            model = YOLO('yolov8n.pt')
            
            # Train the model
            results = model.train(
                data=self.data_yaml_path,
                epochs=30,  # Reduced epochs for faster training
                imgsz=640,
                batch=4,  # Smaller batch for compatibility
                device='cpu',
                project='runs/detect',
                name='coral_detection',
                save_period=10,
                patience=10,
                verbose=True
            )
            
            # Save the best model
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, self.model_path)
                return True, f"Model trained successfully and saved to {self.model_path}"
            else:
                return False, "Training completed but best model not found"
                
        except Exception as e:
            return False, f"Alternative training method failed: {str(e)}"
    
    def create_sample_annotations(self):
        """Create sample annotation format guide"""
        sample_content = """# YOLO Annotation Format
# Each line represents one object: class_id center_x center_y width height
# All values are normalized (0-1)
# For coral class (class_id = 0):

# Example annotation for coral_image_001.txt:
0 0.5 0.3 0.2 0.4
0 0.7 0.6 0.15 0.25

# This represents:
# - Object 1: coral at center (0.5, 0.3) with size (0.2, 0.4)
# - Object 2: coral at center (0.7, 0.6) with size (0.15, 0.25)
"""
        
        with open("data/annotation_guide.txt", "w") as f:
            f.write(sample_content)
