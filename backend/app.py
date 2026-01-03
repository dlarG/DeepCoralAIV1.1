import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
from ultralytics import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://acb-marine.vercel.app",
            "https://*.vercel.app"
        ]
    }
})


port = int(os.environ.get("PORT", 5000))
# Configuration - Match training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
SEGMENTATION_MODEL_PATH = "models/segment/CORAL_segment.pt"
COTS_MODEL_PATH = "models/cots/COTS_counter.pt"

IMG_SIZE = 640  # Match training imgsz=640
CONF_THRESHOLD = 0.25  # Standard YOLO confidence threshold
COTS_CONF_THRESHOLD = 0.5  # Higher confidence for COTS detection

# Create directories
os.makedirs('models/segment', exist_ok=True)
os.makedirs('models/cots', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Load models
def load_segmentation_model():
    try:
        model = YOLO(SEGMENTATION_MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"Failed to load segmentation model: {e}")
        return None

def load_cots_model():
    try:
        model = YOLO(COTS_MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"Failed to load COTS model: {e}")
        return None

# Initialize models
try:
    segmentation_model = load_segmentation_model()
    print("✅ Segmentation model loaded successfully!")
except Exception as e:
    print(f"❌ Segmentation model loading failed: {e}")
    segmentation_model = None

try:
    cots_model = load_cots_model()
    print("✅ COTS model loaded successfully!")
except Exception as e:
    print(f"❌ COTS model loading failed: {e}")
    cots_model = None

# Coral class mapping - Match your training class names exactly
CORAL_CLASSES = {
    0: {"name": "Acropora Branching", "display_name": "Acropora Branching", "color": [255, 0, 0], "category": "hard_coral"},
    1: {"name": "Acropora Tabulate", "display_name": "Acropora Tabulate", "color": [0, 255, 0], "category": "hard_coral"},
    2: {"name": "Encrusting", "display_name": "Encrusting", "color": [0, 0, 255], "category": "hard_coral"},
    3: {"name": "Foliose", "display_name": "Foliose", "color": [255, 255, 0], "category": "hard_coral"},
    4: {"name": "Massive", "display_name": "Massive", "color": [255, 0, 255], "category": "hard_coral"},
    5: {"name": "Non-acropora branching", "display_name": "Non-acropora Branching", "color": [0, 255, 255], "category": "hard_coral"},
    6: {"name": "Submassive", "display_name": "Submassive", "color": [255, 165, 0], "category": "hard_coral"},
    7: {"name": "mushroom", "display_name": "Mushroom", "color": [128, 0, 128], "category": "hard_coral"}
}

# COTS class mapping
COTS_CLASSES = {
    0: {"name": "cots", "display_name": "Crown-of-Thorns Starfish", "color": [255, 0, 0]}
}

def predict_segmentation(image):
    """Perform segmentation on input image using YOLO - match training pipeline"""
    if segmentation_model is None:
        raise Exception("Segmentation model not loaded")
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    # YOLO inference with same settings as training
    results = segmentation_model.predict(
        image_pil, 
        imgsz=IMG_SIZE,  # Match training imgsz=640
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2] if isinstance(image, np.ndarray) else (image_pil.height, image_pil.width)
    
    # Initialize mask with background class
    final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    # Extract segmentation masks
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # Shape: (N, H, W)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class indices
        
        for i, (mask, class_id) in enumerate(zip(masks, classes)):
            # Resize mask to original image size
            mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary mask
            binary_mask = (mask_resized > 0.5).astype(bool)
            
            # Assign class ID to mask (add 1 since YOLO classes start at 0, but we want 1-8 for corals)
            final_mask[binary_mask] = class_id + 1
    
    return final_mask

def predict_cots_detection(image):
    """Perform COTS detection on input image using YOLO11"""
    if cots_model is None:
        raise Exception("COTS model not loaded")
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    # YOLO inference
    results = cots_model.predict(
        image_pil,
        imgsz=IMG_SIZE,
        conf=COTS_CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = box
            detections.append({
                'id': i + 1,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': COTS_CLASSES[cls]['name'],
                'display_name': COTS_CLASSES[cls]['display_name']
            })
    
    return detections

def draw_cots_detections(image, detections):
    """Draw bounding boxes and labels on image"""
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw label
        label = f"COTS #{detection['id']} ({confidence:.2f})"
        
        # Get text size for background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], fill="red")
        
        # Draw label text
        draw.text((x1 + 5, y1 - text_height - 5), label, fill="white", font=font)
    
    return np.array(image_pil)

def create_colored_mask(mask):
    """Convert mask to colored image"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Background stays black (0)
    for class_id, class_info in CORAL_CLASSES.items():
        # Map YOLO class indices (0-7) to mask values (1-8)
        mask_value = class_id + 1
        colored_mask[mask == mask_value] = class_info["color"]
    
    return colored_mask

def create_overlay(image, mask, alpha=0.5):
    """Create overlay of original image and mask"""
    colored_mask = create_colored_mask(mask)
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored_mask, alpha, 0)
    else:
        # Convert grayscale to RGB if needed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        overlay = cv2.addWeighted(image_rgb.astype(np.uint8), 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

def calculate_coral_statistics(mask):
    """Calculate coral coverage statistics"""
    total_pixels = mask.shape[0] * mask.shape[1]
    stats = {}
    coral_pixels = 0
    
    for class_id, class_info in CORAL_CLASSES.items():
        # Map YOLO class indices (0-7) to mask values (1-8)
        mask_value = class_id + 1
        class_pixels = np.sum(mask == mask_value)
        percentage = (class_pixels / total_pixels) * 100
        
        stats[class_info["name"]] = {
            'display_name': class_info["display_name"],
            'pixels': int(class_pixels),
            'percentage': round(percentage, 2),
            'color': class_info["color"],
            'category': class_info["category"]
        }
        
        coral_pixels += class_pixels
    
    # Total coral coverage
    total_coral_percentage = (coral_pixels / total_pixels) * 100
    stats['total_coral'] = {
        'display_name': 'Total Coral Coverage',
        'pixels': int(coral_pixels),
        'percentage': round(total_coral_percentage, 2),
        'color': [0, 128, 0],  # Green for total
        'category': 'summary'
    }
    
    return stats

def calculate_cots_statistics(detections):
    """Calculate COTS detection statistics"""
    total_count = len(detections)
    
    # Group by confidence levels
    high_conf = len([d for d in detections if d['confidence'] >= 0.8])
    medium_conf = len([d for d in detections if 0.5 <= d['confidence'] < 0.8])
    low_conf = len([d for d in detections if d['confidence'] < 0.5])
    
    # Calculate average confidence
    avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
    
    stats = {
        'total_count': {
            'display_name': 'Total COTS Detected',
            'count': total_count,
            'color': [255, 0, 0],
            'category': 'summary'
        },
        'high_confidence': {
            'display_name': 'High Confidence (≥80%)',
            'count': high_conf,
            'color': [255, 0, 0],
            'category': 'confidence'
        },
        'medium_confidence': {
            'display_name': 'Medium Confidence (50-79%)',
            'count': medium_conf,
            'color': [255, 165, 0],
            'category': 'confidence'
        },
        'low_confidence': {
            'display_name': 'Low Confidence (<50%)',
            'count': low_conf,
            'color': [255, 255, 0],
            'category': 'confidence'
        },
        'average_confidence': {
            'display_name': 'Average Confidence',
            'percentage': round(avg_confidence * 100, 2) if detections else 0,
            'color': [0, 255, 0],
            'category': 'summary'
        }
    }
    
    return stats

def image_to_base64(img):
    """Convert image to base64 string"""
    if isinstance(img, np.ndarray):
        # Ensure correct data type and range
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
    
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "device": DEVICE,
        "segmentation_model_loaded": segmentation_model is not None,
        "cots_model_loaded": cots_model is not None,
        "img_size": IMG_SIZE,
        "confidence_threshold": CONF_THRESHOLD,
        "cots_confidence_threshold": COTS_CONF_THRESHOLD
    })

@app.route('/api/segment', methods=['POST'])
def segment_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if segmentation_model is None:
            return jsonify({"error": "Segmentation model not loaded"}), 503
        
        # Read and process image - match training preprocessing
        image_bytes = file.read()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        # Convert BGR to RGB (OpenCV uses BGR, but PIL/YOLO expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform segmentation
        mask = predict_segmentation(image_rgb)
        
        # Create outputs
        colored_mask = create_colored_mask(mask)
        overlay = create_overlay(image_rgb, mask)
        
        # Calculate statistics
        stats = calculate_coral_statistics(mask)
        
        # Convert images to base64
        original_b64 = image_to_base64(image_rgb)
        mask_b64 = image_to_base64(colored_mask)
        overlay_b64 = image_to_base64(overlay)
        
        response = {
            "success": True,
            "statistics": stats,
            "images": {
                "original": f"data:image/png;base64,{original_b64}",
                "mask": f"data:image/png;base64,{mask_b64}",
                "overlay": f"data:image/png;base64,{overlay_b64}"
            },
            "class_info": [
                {
                    "id": class_id,
                    "name": class_info["name"],
                    "display_name": class_info["display_name"],
                    "color": class_info["color"],
                    "category": class_info["category"]
                }
                for class_id, class_info in CORAL_CLASSES.items()
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cots-counter', methods=['POST'])
def cots_counter():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if cots_model is None:
            return jsonify({"error": "COTS model not loaded"}), 503
        
        # Read and process image
        image_bytes = file.read()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        # Convert BGR to RGB (OpenCV uses BGR, but PIL/YOLO expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform COTS detection
        detections = predict_cots_detection(image_rgb)
        
        # Create annotated image with bounding boxes
        annotated_image = draw_cots_detections(image_rgb, detections)
        
        # Calculate statistics
        stats = calculate_cots_statistics(detections)
        
        # Convert images to base64
        original_b64 = image_to_base64(image_rgb)
        annotated_b64 = image_to_base64(annotated_image)
        
        response = {
            "success": True,
            "statistics": stats,
            "detections": detections,
            "images": {
                "original": f"data:image/png;base64,{original_b64}",
                "annotated": f"data:image/png;base64,{annotated_b64}",
                "overlay": f"data:image/png;base64,{annotated_b64}"  # Same as annotated for consistency
            },
            "class_info": [
                {
                    "id": class_id,
                    "name": class_info["name"],
                    "display_name": class_info["display_name"],
                    "color": class_info["color"]
                }
                for class_id, class_info in COTS_CLASSES.items()
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"COTS detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/class-info', methods=['GET'])
def get_class_info():
    """Get information about coral classes"""
    return jsonify({
        "coral_classes": [
            {
                "id": class_id,
                "name": class_info["name"],
                "display_name": class_info["display_name"],
                "color": class_info["color"],
                "category": class_info["category"]
            }
            for class_id, class_info in CORAL_CLASSES.items()
        ],
        "cots_classes": [
            {
                "id": class_id,
                "name": class_info["name"],
                "display_name": class_info["display_name"],
                "color": class_info["color"]
            }
            for class_id, class_info in COTS_CLASSES.items()
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Coral Analysis API...")
    print(f"📊 Segmentation model loaded: {segmentation_model is not None}")
    print(f"⭐ COTS model loaded: {cots_model is not None}")
    print(f"🎯 Device: {DEVICE}")
    print(f"🖼️ Image size: {IMG_SIZE}")
    print(f"🎯 Segmentation confidence threshold: {CONF_THRESHOLD}")
    print(f"⭐ COTS confidence threshold: {COTS_CONF_THRESHOLD}")
    print(f"🪸 Number of coral classes: {len(CORAL_CLASSES)}")
    print(f"⭐ Number of COTS classes: {len(COTS_CLASSES)}")
    app.run(host='0.0.0.0', port=port)