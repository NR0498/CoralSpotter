import cv2
import numpy as np
from PIL import Image
import io
import base64

class ImageUtils:
    @staticmethod
    def pil_to_cv2(pil_image):
        """Convert PIL Image to OpenCV format"""
        # Convert PIL to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        cv2_image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        return cv2_image
    
    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert OpenCV image to PIL format"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    @staticmethod
    def resize_image(image, target_size, maintain_aspect_ratio=True):
        """Resize image to target size"""
        if isinstance(image, Image.Image):
            # PIL Image
            if maintain_aspect_ratio:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                return image
            else:
                return image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # OpenCV image
            if maintain_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calculate scaling factor
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                return cv2.resize(image, (new_w, new_h))
            else:
                return cv2.resize(image, target_size)
    
    @staticmethod
    def normalize_image(image, target_range=(0, 1)):
        """Normalize image values to target range"""
        image = image.astype(np.float32)
        
        # Normalize to 0-1
        image = image / 255.0
        
        # Scale to target range
        if target_range != (0, 1):
            min_val, max_val = target_range
            image = image * (max_val - min_val) + min_val
        
        return image
    
    @staticmethod
    def enhance_underwater_image(image):
        """Apply underwater image enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Color correction for underwater images
        # Reduce blue/green tint
        enhanced_bgr[:, :, 0] = cv2.addWeighted(enhanced_bgr[:, :, 0], 0.8, 
                                               enhanced_bgr[:, :, 1], 0.2, 0)
        
        return enhanced_bgr
    
    @staticmethod
    def create_thumbnail(image, size=(150, 150)):
        """Create thumbnail of image"""
        if isinstance(image, Image.Image):
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail
        else:
            # OpenCV image
            return cv2.resize(image, size)
    
    @staticmethod
    def image_to_base64(image):
        """Convert image to base64 string"""
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        else:
            # OpenCV image
            _, buffer = cv2.imencode('.png', image)
            img_str = base64.b64encode(buffer).decode()
            return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def validate_image(image_data):
        """Validate image data"""
        try:
            if isinstance(image_data, Image.Image):
                # Check image size
                width, height = image_data.size
                if width < 50 or height < 50:
                    return False, "Image too small (minimum 50x50 pixels)"
                
                if width > 4000 or height > 4000:
                    return False, "Image too large (maximum 4000x4000 pixels)"
                
                return True, "Valid image"
            
            elif isinstance(image_data, np.ndarray):
                # OpenCV image
                if len(image_data.shape) < 2:
                    return False, "Invalid image format"
                
                height, width = image_data.shape[:2]
                if width < 50 or height < 50:
                    return False, "Image too small (minimum 50x50 pixels)"
                
                if width > 4000 or height > 4000:
                    return False, "Image too large (maximum 4000x4000 pixels)"
                
                return True, "Valid image"
            
            else:
                return False, "Unsupported image format"
                
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
