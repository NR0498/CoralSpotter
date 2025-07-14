#!/usr/bin/env python3
"""
Train both YOLO detection and coral classification models
"""

import os
import sys
import subprocess
from pathlib import Path

def install_ultralytics():
    """Install ultralytics using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--break-system-packages"])
        print("✓ Ultralytics installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install ultralytics: {e}")
        return False

def train_classification_model():
    """Train the coral classification model"""
    print("\n🔬 Training Coral Classification Model...")
    print("=" * 50)
    
    try:
        from models.simple_classifier import SimpleCoralClassifier
        
        classifier = SimpleCoralClassifier()
        success, message = classifier.train()
        
        if success:
            print(f"✓ Classification model trained successfully!")
            print(f"Details: {message}")
            return True
        else:
            print(f"✗ Classification training failed: {message}")
            return False
            
    except Exception as e:
        print(f"✗ Error training classification model: {e}")
        return False

def train_yolo_model():
    """Train the YOLO detection model"""
    print("\n🎯 Training YOLO Detection Model...")
    print("=" * 50)
    
    # Try to install ultralytics first
    if not install_ultralytics():
        print("Cannot proceed without ultralytics")
        return False
    
    try:
        # Import after installation
        from ultralytics import YOLO
        
        # Check if dataset exists
        train_images = Path("data/images/train")
        train_labels = Path("data/labels/train")
        
        if not train_images.exists() or len(list(train_images.glob("*.jpg"))) == 0:
            print("✗ No training images found")
            return False
        
        if not train_labels.exists() or len(list(train_labels.glob("*.txt"))) == 0:
            print("✗ No training labels found")
            return False
        
        print(f"Found {len(list(train_images.glob('*.jpg')))} training images")
        print(f"Found {len(list(train_labels.glob('*.txt')))} label files")
        
        # Create dataset configuration
        dataset_config = {
            'train': 'data/images/train',
            'val': 'data/images/val',
            'nc': 1,  # number of classes (coral)
            'names': ['coral']
        }
        
        import yaml
        with open("data/coral_dataset.yaml", 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Create validation split if empty
        val_images = Path("data/images/val")
        if len(list(val_images.glob("*.jpg"))) == 0:
            # Move some training images to validation
            train_imgs = list(train_images.glob("*.jpg"))
            val_count = max(1, len(train_imgs) // 5)  # 20% for validation
            
            import shutil
            for img_path in train_imgs[:val_count]:
                # Move image
                shutil.move(str(img_path), str(val_images / img_path.name))
                
                # Move corresponding label if exists
                label_name = img_path.stem + ".txt"
                label_path = train_labels / label_name
                if label_path.exists():
                    shutil.move(str(label_path), str(Path("data/labels/val") / label_name))
        
        # Load pre-trained YOLOv8 model
        print("Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')
        
        # Train the model
        print("Starting training...")
        results = model.train(
            data="data/coral_dataset.yaml",
            epochs=30,  # Reduced for faster training
            imgsz=640,
            batch=4,   # Smaller batch for compatibility
            device='cpu',
            project='runs/detect',
            name='coral_detection',
            save_period=10,
            patience=10,
            verbose=True
        )
        
        # Save the best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        model_save_path = "trained_models/coral_yolo.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, model_save_path)
            print(f"✓ YOLO model saved to {model_save_path}")
            return True
        else:
            print("✗ Training completed but best model not found")
            return False
            
    except Exception as e:
        print(f"✗ Error training YOLO model: {e}")
        return False

def create_dataset_info():
    """Create dataset information file"""
    print("\n📊 Creating Dataset Information...")
    
    dataset_info = {
        "dataset_name": "Coral Reef Detection Dataset",
        "description": "Synthetic coral reef images for detection and classification training",
        "created_by": "Coral Reef AI System",
        "total_images": 0,
        "coral_types": [],
        "yolo_training": {
            "total_images": 0,
            "training_images": 0,
            "validation_images": 0,
            "annotations": 0
        },
        "classification_training": {
            "total_images": 0,
            "coral_types": {}
        }
    }
    
    # Count YOLO images
    train_imgs = list(Path("data/images/train").glob("*.jpg"))
    val_imgs = list(Path("data/images/val").glob("*.jpg"))
    train_labels = list(Path("data/labels/train").glob("*.txt"))
    val_labels = list(Path("data/labels/val").glob("*.txt"))
    
    dataset_info["yolo_training"]["training_images"] = len(train_imgs)
    dataset_info["yolo_training"]["validation_images"] = len(val_imgs)
    dataset_info["yolo_training"]["total_images"] = len(train_imgs) + len(val_imgs)
    dataset_info["yolo_training"]["annotations"] = len(train_labels) + len(val_labels)
    
    # Count classification images
    coral_types_dir = Path("data/coral_types")
    total_classification_images = 0
    coral_types = []
    
    for coral_dir in coral_types_dir.iterdir():
        if coral_dir.is_dir():
            coral_type = coral_dir.name
            coral_images = len(list(coral_dir.glob("*.jpg")))
            coral_types.append(coral_type)
            dataset_info["classification_training"]["coral_types"][coral_type] = coral_images
            total_classification_images += coral_images
    
    dataset_info["classification_training"]["total_images"] = total_classification_images
    dataset_info["coral_types"] = coral_types
    dataset_info["total_images"] = dataset_info["yolo_training"]["total_images"] + total_classification_images
    
    # Save dataset info
    import json
    with open("data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("✓ Dataset information saved to data/dataset_info.json")
    print(f"  Total images: {dataset_info['total_images']}")
    print(f"  YOLO training: {dataset_info['yolo_training']['total_images']} images")
    print(f"  Classification: {dataset_info['classification_training']['total_images']} images")
    print(f"  Coral types: {len(coral_types)}")

def main():
    """Main training function"""
    print("🐠 Coral Reef AI Model Training")
    print("=" * 50)
    
    # Create dataset info
    create_dataset_info()
    
    # Train classification model
    classification_success = train_classification_model()
    
    # Train YOLO model
    yolo_success = train_yolo_model()
    
    # Summary
    print("\n🎯 Training Summary")
    print("=" * 50)
    
    if classification_success:
        print("✓ Coral Classification Model: TRAINED")
    else:
        print("✗ Coral Classification Model: FAILED")
    
    if yolo_success:
        print("✓ YOLO Detection Model: TRAINED")
    else:
        print("✗ YOLO Detection Model: FAILED")
    
    if classification_success and yolo_success:
        print("\n🚀 All models trained successfully!")
        print("The coral reef detection system is ready to use.")
    else:
        print("\n⚠️  Some models failed to train.")
        print("Check the error messages above for details.")
    
    return classification_success and yolo_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)