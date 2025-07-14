import os
import shutil
import yaml
from pathlib import Path
import json

class RobustYOLOTrainer:
    def __init__(self):
        self.model_path = "trained_models/coral_yolo.pt"
        self.config_path = "data/coral_dataset.yaml"
        self.project_dir = "runs/detect"
        
    def install_ultralytics(self):
        """Install ultralytics if not available"""
        try:
            import ultralytics
            return True
        except ImportError:
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                return True
            except:
                return False
    
    def validate_dataset(self):
        """Validate that we have sufficient training data"""
        train_images = Path("data/images/train")
        train_labels = Path("data/labels/train")
        val_images = Path("data/images/val")
        val_labels = Path("data/labels/val")
        
        if not all([train_images.exists(), train_labels.exists(), val_images.exists(), val_labels.exists()]):
            return False, "Missing required directories"
        
        train_img_count = len(list(train_images.glob("*.jpg")))
        train_label_count = len(list(train_labels.glob("*.txt")))
        val_img_count = len(list(val_images.glob("*.jpg")))
        val_label_count = len(list(val_labels.glob("*.txt")))
        
        if train_img_count < 10:
            return False, f"Insufficient training images: {train_img_count} (minimum: 10)"
        
        if train_img_count != train_label_count:
            return False, f"Mismatch: {train_img_count} images vs {train_label_count} labels"
        
        if val_img_count != val_label_count:
            return False, f"Validation mismatch: {val_img_count} images vs {val_label_count} labels"
        
        return True, f"Dataset validated: {train_img_count} train, {val_img_count} val images"
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration"""
        config = {
            'names': ['coral'],
            'nc': 1,
            'train': 'data/images/train',
            'val': 'data/images/val'
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    
    def train_model(self, epochs=30, batch_size=8, img_size=640):
        """Train YOLO model with proper error handling"""
        try:
            # Install ultralytics if needed
            if not self.install_ultralytics():
                return False, "Failed to install ultralytics"
            
            from ultralytics import YOLO
            
            # Validate dataset
            valid, message = self.validate_dataset()
            if not valid:
                return False, f"Dataset validation failed: {message}"
            
            # Create dataset config
            self.create_dataset_config()
            
            # Initialize model
            model = YOLO('yolov8n.pt')  # Use pretrained model
            
            # Train the model
            results = model.train(
                data=self.config_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=self.project_dir,
                name='coral_training',
                exist_ok=True,
                save=True,
                verbose=True
            )
            
            # Copy the best model to our model directory
            best_model_path = Path(self.project_dir) / 'coral_training' / 'weights' / 'best.pt'
            if best_model_path.exists():
                Path("trained_models").mkdir(exist_ok=True)
                shutil.copy2(best_model_path, self.model_path)
                
                # Save training info
                training_info = {
                    "model_type": "YOLOv8n",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "image_size": img_size,
                    "dataset": message,
                    "model_path": str(self.model_path),
                    "training_complete": True
                }
                
                with open("trained_models/yolo_training_info.json", "w") as f:
                    json.dump(training_info, f, indent=2)
                
                return True, f"YOLO model trained successfully! {message}"
            else:
                return False, "Training completed but model file not found"
                
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def test_model(self, test_image_path):
        """Test the trained model on an image"""
        try:
            if not os.path.exists(self.model_path):
                return False, "Model not found. Please train first."
            
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            
            results = model(test_image_path)
            return True, f"Model test successful on {test_image_path}"
            
        except Exception as e:
            return False, f"Model test failed: {str(e)}"