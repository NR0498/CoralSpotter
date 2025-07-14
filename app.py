import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from utils.visualization import draw_detections
import tempfile

# Delayed import to avoid TensorFlow loading issues
CoralDetectionInference = None

# Configure page
st.set_page_config(
    page_title="Coral Reef Detection & Classification",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None

def load_models():
    """Load the trained models"""
    try:
        if st.session_state.inference_engine is None:
            with st.spinner("Loading AI models..."):
                # Delayed import to avoid initial loading issues
                global CoralDetectionInference
                if CoralDetectionInference is None:
                    from models.inference import CoralDetectionInference
                st.session_state.inference_engine = CoralDetectionInference()
        return True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure both models are trained before analyzing images.")
        return False

def main():
    st.title("🐠 Coral Reef Detection & Classification System")
    st.markdown("Upload an underwater image to detect and classify coral reef lifeforms")
    
    # Sidebar for model training and information
    with st.sidebar:
        st.header("🔧 Model Management")
        
        st.subheader("YOLO Detection Model")
        if st.button("Train YOLO Model", type="primary"):
            train_yolo_model()
        
        st.subheader("Coral Classification Model")
        if st.button("Train Classification Model", type="primary"):
            train_classification_model()
        
        st.subheader("Model Information")
        check_model_status()
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Requirements")
        st.markdown("- **YOLO Training**: 30+ annotated images")
        st.markdown("- **Classification**: Multiple coral species")
        st.markdown("- **Formats**: JPG, PNG, JPEG")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an underwater image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect and classify coral reefs"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                analyze_image(image, col2)
    
    with col2:
        st.header("🎯 Detection Results")
        st.info("Upload an image and click 'Analyze Image' to see results")

def train_yolo_model():
    """Train YOLO model for coral detection"""
    with st.spinner("Training YOLO model..."):
        try:
            from models.yolo_trainer import YOLOTrainer
            trainer = YOLOTrainer()
            
            # Check if training data exists
            if not os.path.exists("data/images/train") or len(os.listdir("data/images/train")) == 0:
                st.error("No training images found. Please add images to data/images/train/")
                return
            
            success, message = trainer.train()
            if success:
                st.success("YOLO model trained successfully!")
                st.balloons()
            else:
                st.error(f"Training failed: {message}")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

def train_classification_model():
    """Train classification model for coral species"""
    with st.spinner("Training coral classification model..."):
        try:
            from models.simple_classifier import SimpleCoralClassifier
            classifier = SimpleCoralClassifier()
            
            # Check if classification data exists
            if not os.path.exists("data/coral_types") or len(os.listdir("data/coral_types")) == 0:
                st.error("No coral classification data found. Please add coral type folders to data/coral_types/")
                return
            
            success, message = classifier.train()
            if success:
                st.success("Coral classification model trained successfully!")
                st.balloons()
                st.info("Model uses Random Forest classifier with computer vision features")
            else:
                st.error(f"Training failed: {message}")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

def check_model_status():
    """Check if models are trained and available"""
    yolo_path = "trained_models/coral_yolo.pt"
    classifier_path = "trained_models/coral_classifier.pkl"
    
    st.markdown("**YOLO Model:**")
    if os.path.exists(yolo_path):
        st.success("✅ Trained")
    else:
        st.warning("❌ Not trained")
    
    st.markdown("**Classifier Model:**")
    if os.path.exists(classifier_path):
        st.success("✅ Trained (Random Forest)")
        # Show dataset info if available
        if os.path.exists("data/dataset_info.json"):
            import json
            with open("data/dataset_info.json", "r") as f:
                dataset_info = json.load(f)
            st.text(f"Dataset: {dataset_info['classification_training']['total_images']} images")
            st.text(f"Types: {len(dataset_info['coral_types'])} coral species")
    else:
        st.warning("❌ Not trained")

def analyze_image(image, result_column):
    """Analyze uploaded image for coral detection and classification"""
    if not load_models():
        return
    
    with result_column:
        with st.spinner("Analyzing image..."):
            try:
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Run inference
                results = st.session_state.inference_engine.predict(img_array)
                
                if results['coral_detected']:
                    st.success(f"🐠 Coral Detected! ({len(results['detections'])} regions)")
                    
                    # Display detection details
                    for i, detection in enumerate(results['detections']):
                        st.markdown(f"**Detection {i+1}:**")
                        st.markdown(f"- Confidence: {detection['confidence']:.2%}")
                        if detection['coral_type']:
                            st.markdown(f"- Coral Type: {detection['coral_type']}")
                            st.markdown(f"- Classification Confidence: {detection['classification_confidence']:.2%}")
                    
                    # Display annotated image
                    annotated_image = draw_detections(img_array, results['detections'])
                    st.image(annotated_image, caption="Detection Results", use_column_width=True)
                    
                    # Statistics
                    st.markdown("### 📊 Analysis Summary")
                    total_detections = len(results['detections'])
                    classified_corals = sum(1 for d in results['detections'] if d['coral_type'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Detections", total_detections)
                    with col2:
                        st.metric("Classified Corals", classified_corals)
                    
                else:
                    st.warning("🌊 No coral reef lifeforms detected in this image")
                    st.info("Try uploading an underwater image with visible coral formations")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please ensure both models are trained before analyzing images")

if __name__ == "__main__":
    main()
