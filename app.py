import streamlit as st
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="🐠 Coral Reef Detection & Classification System",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced dark mode interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .upload-box {
        border: 2px dashed #00d4ff;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        background: #262730;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #33e6ff;
        background: #2d2e3f;
    }
    
    .detection-result {
        background: #262730;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #00d4ff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-section {
        background: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00d4ff;
    }
    
    .status-success {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff9500;
        font-weight: bold;
    }
    
    .coral-type-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: #00d4ff;
        color: #0e1117;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None

def load_models():
    """Load the trained models"""
    try:
        if st.session_state.inference_engine is None:
            with st.spinner("🤖 Loading AI models..."):
                from models.inference import CoralDetectionInference
                st.session_state.inference_engine = CoralDetectionInference()
        return True
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("💡 Make sure both models are trained before analyzing images.")
        return False

def train_yolo_model():
    """Train YOLO model using robust trainer"""
    with st.spinner("🚀 Training YOLO Detection Model..."):
        try:
            from models.robust_yolo_trainer import RobustYOLOTrainer
            trainer = RobustYOLOTrainer()
            
            success, message = trainer.train_model(epochs=30, batch_size=8)
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
                st.info("🎯 YOLO model can now detect coral regions in underwater images!")
            else:
                st.error(f"❌ Training failed: {message}")
                st.info("💡 Check dataset requirements and try again")
                
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")

def train_classification_model():
    """Train classification model for coral species"""
    with st.spinner("🧠 Training Coral Classification Model..."):
        try:
            from models.simple_classifier import SimpleCoralClassifier
            classifier = SimpleCoralClassifier()
            
            # Check if classification data exists
            if not os.path.exists("data/coral_types") or len(os.listdir("data/coral_types")) == 0:
                st.error("❌ No coral classification data found. Please add coral type folders to data/coral_types/")
                return
            
            success, message = classifier.train()
            if success:
                st.success("✅ Coral classification model trained successfully!")
                st.balloons()
                st.info("🔬 Model uses Random Forest classifier with computer vision features")
            else:
                st.error(f"❌ Training failed: {message}")
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")

def check_model_status():
    """Check if models are trained and available"""
    st.markdown("### 📊 Model Status")
    
    yolo_path = "trained_models/coral_yolo.pt"
    classifier_path = "trained_models/coral_classifier.pkl"
    
    # YOLO Model Status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**🎯 YOLO Detection Model:**")
    with col2:
        if os.path.exists(yolo_path):
            st.markdown('<span class="status-success">✅ Trained</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">❌ Not trained</span>', unsafe_allow_html=True)
    
    # Classification Model Status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**🧠 Classification Model:**")
    with col2:
        if os.path.exists(classifier_path):
            st.markdown('<span class="status-success">✅ Trained</span>', unsafe_allow_html=True)
            
            # Show dataset info if available
            if os.path.exists("data/dataset_info.json"):
                with open("data/dataset_info.json", "r") as f:
                    dataset_info = json.load(f)
                st.caption(f"📸 Dataset: {dataset_info.get('classification', {}).get('total_images', 'N/A')} images")
                st.caption(f"🐠 Species: {len(dataset_info.get('classification', {}).get('coral_types', []))} coral types")
        else:
            st.markdown('<span class="status-warning">❌ Not trained</span>', unsafe_allow_html=True)

def display_dataset_management():
    """Display dataset management interface"""
    st.markdown("### 📁 Dataset Management")
    
    try:
        from utils.dataset_manager import DatasetManager
        dataset_manager = DatasetManager()
        dataset_manager.setup_directories()
        
        # Dataset statistics
        stats = dataset_manager.get_dataset_stats()
        
        st.markdown("#### 📊 Current Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 YOLO Detection:**")
            st.write(f"Training: {stats['yolo']['train_images']} images")
            st.write(f"Validation: {stats['yolo']['val_images']} images")
        
        with col2:
            st.markdown("**🔬 Classification:**")
            total_class_images = sum(stats['classification'].values())
            st.write(f"Total: {total_class_images} images")
            st.write(f"Species: {len([k for k, v in stats['classification'].items() if v > 0])}")
        
        # Add new data section
        with st.expander("➕ Add New Training Data"):
            tab1, tab2 = st.tabs(["🎯 YOLO Detection", "🔬 Classification"])
            
            with tab1:
                st.markdown("**Add YOLO Training Image**")
                uploaded_file = st.file_uploader("Upload underwater image", type=['jpg', 'jpeg', 'png'], key="yolo_upload")
                
                if uploaded_file:
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    
                    st.markdown("**YOLO Annotation:**")
                    example_annotation = dataset_manager.create_annotation_example()
                    annotation_text = st.text_area(
                        "Enter YOLO annotations",
                        value="0 0.5 0.5 0.3 0.4",
                        help="Format: class_id center_x center_y width height (normalized 0-1)",
                        height=100
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        split = st.selectbox("Dataset split", ["train", "val"])
                    with col2:
                        if st.button("Add to YOLO Dataset"):
                            valid, message = dataset_manager.validate_yolo_annotation(annotation_text)
                            if valid:
                                success, result = dataset_manager.add_yolo_image(uploaded_file, annotation_text, split)
                                if success:
                                    st.success(result)
                                    dataset_manager.update_dataset_info()
                                else:
                                    st.error(result)
                            else:
                                st.error(f"Invalid annotation: {message}")
            
            with tab2:
                st.markdown("**Add Classification Image**")
                class_uploaded_file = st.file_uploader("Upload coral image", type=['jpg', 'jpeg', 'png'], key="class_upload")
                
                if class_uploaded_file:
                    st.image(class_uploaded_file, caption="Uploaded Coral", use_column_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        coral_type = st.selectbox("Coral Type", dataset_manager.coral_types)
                    with col2:
                        if st.button("Add to Classification Dataset"):
                            success, result = dataset_manager.add_classification_image(class_uploaded_file, coral_type)
                            if success:
                                st.success(result)
                                dataset_manager.update_dataset_info()
                            else:
                                st.error(result)
        
    except Exception as e:
        st.error(f"Dataset management error: {str(e)}")

def analyze_image(image, result_column):
    """Analyze uploaded image for coral detection and classification"""
    if load_models():
        try:
            with st.spinner("🔍 Analyzing underwater image..."):
                # Convert PIL to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Run inference
                detections = st.session_state.inference_engine.predict(opencv_image)
                
                with result_column:
                    st.markdown('<div class="detection-result">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Detection Results")
                    
                    if detections and len(detections) > 0:
                        st.success(f"✅ Found {len(detections)} coral region(s)")
                        
                        # Draw detections on image
                        from utils.visualization import draw_detections
                        result_image = draw_detections(opencv_image.copy(), detections)
                        
                        # Convert back to RGB for display
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image_rgb, caption="🐠 Detection Results", use_column_width=True)
                        
                        # Display detailed results
                        st.markdown("### 📋 Detailed Analysis")
                        for i, detection in enumerate(detections):
                            with st.expander(f"🐠 Coral Region {i+1}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Detection Confidence", f"{detection.get('detection_confidence', 0):.2%}")
                                with col2:
                                    coral_type = detection.get('coral_type', 'Unknown')
                                    class_confidence = detection.get('classification_confidence', 0)
                                    st.metric("Species", coral_type)
                                    st.metric("Classification Confidence", f"{class_confidence:.2%}")
                                
                                # Display coral type with badge styling
                                if coral_type != 'Unknown':
                                    st.markdown(f'<span class="coral-type-badge">{coral_type.replace("_", " ").title()}</span>', 
                                              unsafe_allow_html=True)
                    else:
                        st.warning("🔍 No coral regions detected in this image")
                        st.info("💡 Try uploading a clearer underwater image with visible coral formations")
                        st.image(image, caption="Original Image", use_column_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            with result_column:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Make sure both models are trained and try again")

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🐠 Coral Reef Detection & Classification System</h1>
        <p>Advanced AI-powered coral identification for marine research and conservation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model management
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("## 🔧 Model Management")
        
        # Training buttons
        st.markdown("### 🚀 Model Training")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Train YOLO", type="primary", use_container_width=True):
                train_yolo_model()
        with col2:
            if st.button("🧠 Train Classifier", type="primary", use_container_width=True):
                train_classification_model()
        
        st.markdown("---")
        check_model_status()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        display_dataset_management()
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📋 System Info")
        st.info("🌊 **Coral Types Supported:**\n" +
                "• Brain Coral\n• Staghorn Coral\n• Table Coral\n" +
                "• Soft Coral\n• Fan Coral\n• Mushroom Coral")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📸 Image Analysis")
        
        # File uploader with enhanced styling
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "🌊 Upload an underwater image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload clear underwater images with coral formations for best results"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                analyze_image(image, col2)
        else:
            st.info("👆 Upload an underwater image to start coral detection and classification")
    
    with col2:
        st.markdown("## 🎯 Analysis Results")
        if uploaded_file is None:
            st.markdown("""
            <div class="detection-result">
                <h3>🌊 Welcome to Coral Reef AI</h3>
                <p>This advanced system uses:</p>
                <ul>
                    <li><strong>🎯 YOLO Detection</strong> - Locates coral regions</li>
                    <li><strong>🧠 Classification AI</strong> - Identifies coral species</li>
                    <li><strong>📊 Confidence Scoring</strong> - Provides accuracy metrics</li>
                </ul>
                <p>Upload an underwater image to begin analysis!</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()