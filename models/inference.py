import os
import numpy as np
import cv2
import json

# Try to import ultralytics, fallback to basic implementation if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    import subprocess
    import sys

# Import simple classifier and detector
try:
    from models.simple_classifier import SimpleCoralClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

try:
    from models.simple_yolo import SimpleCoralDetector
    SIMPLE_DETECTOR_AVAILABLE = True
except ImportError:
    SIMPLE_DETECTOR_AVAILABLE = False

class CoralDetectionInference:
    def __init__(self):
        self.yolo_model_path = "trained_models/coral_yolo.pt"
        self.classifier_model_path = "trained_models/coral_classifier.pkl"
        self.label_encoder_path = "trained_models/coral_labels.json"
        
        self.yolo_model = None
        self.classifier_model = None
        self.label_mapping = None
        
        self.load_models()
    
    def load_models(self):
        """Load both YOLO and classification models"""
        try:
            # Load YOLO model
            if os.path.exists(self.yolo_model_path):
                if YOLO_AVAILABLE:
                    self.yolo_model = YOLO(self.yolo_model_path)
                    print("YOLO model loaded successfully")
                else:
                    # Try to install ultralytics and reload
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                        from ultralytics import YOLO
                        self.yolo_model = YOLO(self.yolo_model_path)
                        print("YOLO model loaded successfully after installation")
                    except:
                        print("YOLO model found but ultralytics not available")
            else:
                # Use simple detector as fallback
                if SIMPLE_DETECTOR_AVAILABLE:
                    self.yolo_model = SimpleCoralDetector()
                    print("Using simple coral detector (demo mode)")
                else:
                    print("No coral detection model available")
            
            # Load classification model
            if os.path.exists(self.classifier_model_path) and CLASSIFIER_AVAILABLE:
                self.classifier_model = SimpleCoralClassifier()
                if self.classifier_model.load_model():
                    print("Classification model loaded successfully")
                else:
                    print("Failed to load classification model")
                    self.classifier_model = None
            else:
                print("Classification model not found. Please train the model first.")
            
            # Load label mapping
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'r') as f:
                    self.label_mapping = json.load(f)
                    # Convert string keys to integers
                    self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
                print("Label mapping loaded successfully")
            else:
                print("Label mapping not found.")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def detect_coral(self, image):
        """Detect coral using YOLO model"""
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image, conf=0.25)  # Confidence threshold
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in coral detection: {e}")
            return []
    
    def classify_coral(self, image_crop):
        """Classify coral type using trained model"""
        if self.classifier_model is None:
            return None, 0.0
        
        try:
            # Use the simple classifier's predict method
            coral_type, confidence = self.classifier_model.predict(image_crop)
            return coral_type, confidence
            
        except Exception as e:
            print(f"Error in coral classification: {e}")
            return None, 0.0
    
    def predict(self, image):
        """Complete prediction pipeline"""
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV operations
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Detect coral regions
        detections = self.detect_coral(image_bgr)
        
        # Process each detection
        processed_detections = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract coral region
            coral_crop = image_bgr[y1:y2, x1:x2]
            
            # Classify coral type if crop is valid
            coral_type = None
            classification_confidence = 0.0
            
            if coral_crop.size > 0:
                coral_type, classification_confidence = self.classify_coral(coral_crop)
            
            processed_detections.append({
                'bbox': bbox,
                'confidence': detection['confidence'],
                'coral_type': coral_type,
                'classification_confidence': classification_confidence
            })
        
        return {
            'coral_detected': len(detections) > 0,
            'detections': processed_detections,
            'total_detections': len(detections)
        }
    
    def is_ready(self):
        """Check if both models are loaded and ready"""
        return (self.yolo_model is not None and 
                self.classifier_model is not None and 
                self.label_mapping is not None)
