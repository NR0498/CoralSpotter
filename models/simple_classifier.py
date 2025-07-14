import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import json

class SimpleCoralClassifier:
    def __init__(self):
        self.model_path = "trained_models/coral_classifier.pkl"
        self.label_encoder_path = "trained_models/coral_labels.json"
        self.feature_size = 64  # Size to resize images for feature extraction
        self.label_encoder = LabelEncoder()
        self.model = None
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("trained_models", exist_ok=True)
        os.makedirs("data/coral_types", exist_ok=True)
        
    def extract_features(self, image):
        """Extract simple features from image"""
        # Resize image
        resized = cv2.resize(image, (self.feature_size, self.feature_size))
        
        # Convert to different color spaces and extract statistics
        features = []
        
        # RGB statistics
        for channel in range(3):
            channel_data = resized[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # HSV statistics
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            channel_data = hsv[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # Texture features (simplified)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.feature_size * self.feature_size)
        features.append(edge_density)
        
        # Local Binary Pattern approximation
        lbp_mean = np.mean(gray)
        lbp_std = np.std(gray)
        features.extend([lbp_mean, lbp_std])
        
        return np.array(features)
    
    def load_coral_data(self):
        """Load coral classification data"""
        data_dir = Path("data/coral_types")
        features = []
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
                        # Extract features
                        feature_vector = self.extract_features(img)
                        features.append(feature_vector)
                        labels.append(coral_type)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if len(features) == 0:
            return None, None, "No valid images found in coral type directories"
        
        return np.array(features), np.array(labels), f"Loaded {len(features)} images from {len(coral_types)} coral types"
    
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
            features, labels, message = self.load_coral_data()
            if features is None:
                return False, message
            
            print(message)
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            print(f"Number of classes: {num_classes}")
            print(f"Classes: {self.label_encoder.classes_}")
            
            # Save label encoder
            self.save_label_encoder(self.label_encoder.classes_)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Create and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            print("Training Random Forest classifier...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"Validation accuracy: {accuracy:.2%}")
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            return True, f"Model trained successfully! Validation accuracy: {accuracy:.2%}"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, image):
        """Predict coral type from image"""
        if self.model is None:
            if not self.load_model():
                return None, 0.0
        
        try:
            # Extract features
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
            
            # Get class name
            label_mapping = self.load_label_encoder()
            if label_mapping:
                coral_type = label_mapping.get(prediction, "Unknown")
            else:
                coral_type = f"Class_{prediction}"
            
            return coral_type, confidence
            
        except Exception as e:
            print(f"Error in coral classification: {e}")
            return None, 0.0