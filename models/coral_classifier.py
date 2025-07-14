import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from pathlib import Path
import json

class CoralClassifier:
    def __init__(self):
        self.model_path = "trained_models/coral_classifier.h5"
        self.label_encoder_path = "trained_models/coral_labels.json"
        self.img_size = (224, 224)
        self.num_classes = 0
        self.label_encoder = LabelEncoder()
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("trained_models", exist_ok=True)
        os.makedirs("data/coral_types", exist_ok=True)
        
    def load_coral_data(self):
        """Load coral classification data"""
        data_dir = Path("data/coral_types")
        images = []
        labels = []
        
        # Get all coral type directories
        coral_types = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if len(coral_types) == 0:
            return None, None, "No coral type directories found in data/coral_types/"
        
        print(f"Found {len(coral_types)} coral types: {[d.name for d in coral_types]}")
        
        for coral_type_dir in coral_types:
            coral_type = coral_type_dir.name
            image_files = list(coral_type_dir.glob("*.jpg")) + \
                         list(coral_type_dir.glob("*.jpeg")) + \
                         list(coral_type_dir.glob("*.png"))
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(coral_type)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            return None, None, "No valid images found in coral type directories"
        
        return np.array(images), np.array(labels), f"Loaded {len(images)} images from {len(coral_types)} coral types"
    
    def create_model(self, num_classes):
        """Create CNN model for coral classification"""
        # Create a simple CNN model for coral classification
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_label_encoder(self, classes):
        """Save label encoder mapping"""
        label_mapping = {str(i): class_name for i, class_name in enumerate(classes)}
        with open(self.label_encoder_path, 'w') as f:
            json.dump(label_mapping, f)
    
    def load_label_encoder(self):
        """Load label encoder mapping"""
        try:
            with open(self.label_encoder_path, 'r') as f:
                label_mapping = json.load(f)
            return {int(k): v for k, v in label_mapping.items()}
        except:
            return None
    
    def train(self):
        """Train coral classification model"""
        try:
            # Load data
            images, labels, message = self.load_coral_data()
            if images is None:
                return False, message
            
            print(message)
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.num_classes = len(self.label_encoder.classes_)
            
            print(f"Number of classes: {self.num_classes}")
            print(f"Classes: {self.label_encoder.classes_}")
            
            # Save label encoder
            self.save_label_encoder(self.label_encoder.classes_)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Create model
            model = self.create_model(self.num_classes)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                keras.callbacks.ModelCheckpoint(
                    self.model_path, 
                    save_best_only=True, 
                    monitor='val_accuracy'
                )
            ]
            
            # Simple training without data augmentation for now
            # (Data augmentation can be added later if needed)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=16,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            return True, f"Model trained successfully! Validation accuracy: {val_accuracy:.2%}"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def create_sample_dataset_structure(self):
        """Create sample dataset structure guide"""
        sample_structure = """# Coral Classification Dataset Structure

data/coral_types/
├── brain_coral/
│   ├── brain_coral_001.jpg
│   ├── brain_coral_002.jpg
│   └── ...
├── staghorn_coral/
│   ├── staghorn_001.jpg
│   ├── staghorn_002.jpg
│   └── ...
├── table_coral/
│   ├── table_coral_001.jpg
│   ├── table_coral_002.jpg
│   └── ...
└── soft_coral/
    ├── soft_coral_001.jpg
    ├── soft_coral_002.jpg
    └── ...

# Instructions:
1. Create folders for each coral type
2. Add at least 10-20 images per coral type
3. Use clear, well-lit underwater photos
4. Ensure images show distinct coral characteristics
"""
        
        with open("data/dataset_structure_guide.txt", "w") as f:
            f.write(sample_structure)
