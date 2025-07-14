import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import io

class VisualizationUtils:
    def __init__(self):
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
    
    def get_color(self, class_id):
        """Get color for a specific class"""
        return self.colors[class_id % len(self.colors)]

def draw_detections(image, detections, show_confidence=True, show_labels=True):
    """Draw detection bounding boxes on image"""
    # Create a copy of the image
    result_image = image.copy()
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        coral_type = detection.get('coral_type', 'Coral')
        classification_confidence = detection.get('classification_confidence', 0.0)
        
        x1, y1, x2, y2 = bbox
        
        # Choose color based on detection index
        color = (0, 255, 0) if coral_type else (255, 0, 0)  # Green if classified, red if not
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_parts = []
        if show_labels and coral_type:
            label_parts.append(coral_type)
        
        if show_confidence:
            label_parts.append(f"{confidence:.2%}")
            if coral_type and classification_confidence > 0:
                label_parts.append(f"({classification_confidence:.2%})")
        
        label_text = " ".join(label_parts)
        
        if label_text:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(
                result_image,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y + 5),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result_image,
                label_text,
                (x1 + 2, label_y - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )
    
    return result_image

def create_detection_summary_plot(detections):
    """Create a summary plot of detections"""
    if not detections:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No coral detections found', 
                ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Count coral types
    coral_types = {}
    confidence_scores = []
    
    for detection in detections:
        coral_type = detection.get('coral_type', 'Unclassified')
        confidence = detection['confidence']
        
        coral_types[coral_type] = coral_types.get(coral_type, 0) + 1
        confidence_scores.append(confidence)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Coral type distribution
    if coral_types:
        types = list(coral_types.keys())
        counts = list(coral_types.values())
        
        ax1.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Coral Type Distribution')
    
    # Plot 2: Detection confidence histogram
    ax2.hist(confidence_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Number of Detections')
    ax2.set_title('Detection Confidence Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection statistics
    stats_data = [
        ['Total Detections', len(detections)],
        ['Avg Confidence', f"{np.mean(confidence_scores):.2%}"],
        ['Min Confidence', f"{np.min(confidence_scores):.2%}"],
        ['Max Confidence', f"{np.max(confidence_scores):.2%}"],
        ['Unique Coral Types', len(coral_types)]
    ]
    
    table = ax3.table(cellText=stats_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.axis('off')
    ax3.set_title('Detection Statistics')
    
    # Plot 4: Confidence vs coral type
    if len(coral_types) > 1:
        type_confidences = {}
        for detection in detections:
            coral_type = detection.get('coral_type', 'Unclassified')
            confidence = detection['confidence']
            
            if coral_type not in type_confidences:
                type_confidences[coral_type] = []
            type_confidences[coral_type].append(confidence)
        
        # Box plot
        box_data = [type_confidences[t] for t in type_confidences.keys()]
        ax4.boxplot(box_data, labels=list(type_confidences.keys()))
        ax4.set_ylabel('Confidence Score')
        ax4.set_title('Confidence by Coral Type')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for comparison', 
                ha='center', va='center')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    
    plt.tight_layout()
    return fig

def create_training_progress_plot(history_data):
    """Create training progress visualization"""
    if not history_data:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(history_data['loss']) + 1)
    
    # Training loss
    ax1.plot(epochs, history_data['loss'], 'b-', label='Training Loss')
    if 'val_loss' in history_data:
        ax1.plot(epochs, history_data['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training accuracy
    if 'accuracy' in history_data:
        ax2.plot(epochs, history_data['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in history_data:
            ax2.plot(epochs, history_data['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    
    # Learning rate (if available)
    if 'lr' in history_data:
        ax3.plot(epochs, history_data['lr'], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
    
    # Training metrics summary
    final_metrics = []
    if 'loss' in history_data:
        final_metrics.append(['Final Training Loss', f"{history_data['loss'][-1]:.4f}"])
    if 'val_loss' in history_data:
        final_metrics.append(['Final Validation Loss', f"{history_data['val_loss'][-1]:.4f}"])
    if 'accuracy' in history_data:
        final_metrics.append(['Final Training Accuracy', f"{history_data['accuracy'][-1]:.2%}"])
    if 'val_accuracy' in history_data:
        final_metrics.append(['Final Validation Accuracy', f"{history_data['val_accuracy'][-1]:.2%}"])
    
    if final_metrics:
        table = ax4.table(cellText=final_metrics,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    ax4.axis('off')
    ax4.set_title('Final Training Metrics')
    
    plt.tight_layout()
    return fig
