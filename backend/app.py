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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEGMENTATION_MODEL_PATH = "models/segment/CORAL_segment.pt"
COTS_MODEL_PATH = "models/cots/COTS_counter.pt"
AUTO_CROP_MODEL_PATH = "models/crop/best.pt" 

IMG_SIZE = 640  
CONF_THRESHOLD = 0.25  
COTS_CONF_THRESHOLD = 0.5  
CROP_CONF_THRESHOLD = 0.25 

os.makedirs('models/segment', exist_ok=True)
os.makedirs('models/cots', exist_ok=True)
os.makedirs('models/crop', exist_ok=True)  
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

def load_auto_crop_model():
    try:
        model = YOLO(AUTO_CROP_MODEL_PATH)
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"Failed to load auto-crop model: {e}")
        return None

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

try:
    auto_crop_model = load_auto_crop_model()
    print("✅ Auto-crop model loaded successfully!")
except Exception as e:
    print(f"❌ Auto-crop model loading failed: {e}")
    auto_crop_model = None

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

COTS_CLASSES = {
    0: {"name": "cots", "display_name": "Crown-of-Thorns Starfish", "color": [255, 0, 0]}
}

def order_points(pts):
    """Order points starting from top-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts, width=None, height=None):
    """Apply perspective transform to rectify quadrat"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    if width is None or height is None:
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        width = maxWidth if width is None else width
        height = maxHeight if height is None else height
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def enhance_contour_detection(mask):
    """Enhance mask for better contour detection"""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def auto_crop_quadrats(image, conf_threshold=0.25):
    """
    Detect and rectify quadrats in the image using auto-crop model.
    Returns list of cropped quadrat images and their bounding boxes.
    """
    if auto_crop_model is None:
        print("Auto-crop model not loaded, returning original image")
        return [image], []
    
    try:
        results = auto_crop_model(image, conf=conf_threshold, verbose=False)
        
        cropped_images = []
        crop_info = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            
            print(f"Found {len(masks)} quadrats in image")
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                try:
                    x1, y1, x2, y2, conf, cls = box
                    
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))
                    mask_enhanced = enhance_contour_detection(mask_resized)
                    
                    contours, _ = cv2.findContours(mask_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        points = None
                        
                        if len(approx) == 4:
                            points = approx.reshape(4, 2).astype(np.float32)
                        elif len(approx) > 4:
                            hull = cv2.convexHull(largest_contour)
                            epsilon = 0.01 * cv2.arcLength(hull, True)
                            approx_hull = cv2.approxPolyDP(hull, epsilon, True)
                            
                            if len(approx_hull) == 4:
                                points = approx_hull.reshape(4, 2).astype(np.float32)
                        
                        if points is None:
                            rect = cv2.minAreaRect(largest_contour)
                            points = cv2.boxPoints(rect)
                            points = np.float32(points)
                        
                        rectified = four_point_transform(image, points)
                        
                        if rectified.shape[0] > 10 and rectified.shape[1] > 10:
                            cropped_images.append(rectified)
                            crop_info.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'points': points.tolist()
                            })
                            print(f"Successfully cropped quadrat {i+1}, shape: {rectified.shape}")
                            
                except Exception as e:
                    print(f"Error processing quadrat {i}: {str(e)}")
                    continue
        
        if not cropped_images:
            print("No quadrats detected, using full image")
            return [image], []
        
        return cropped_images, crop_info
        
    except Exception as e:
        print(f"Error in auto_crop_quadrats: {str(e)}")
        return [image], []

def predict_segmentation(image):
    """Perform segmentation on input image using YOLO - match training pipeline"""
    if segmentation_model is None:
        raise Exception("Segmentation model not loaded")
    
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    results = segmentation_model.predict(
        image_pil, 
        imgsz=IMG_SIZE, 
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    
    orig_height, orig_width = image.shape[:2] if isinstance(image, np.ndarray) else (image_pil.height, image_pil.width)
    
    final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # Shape: (N, H, W)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Class indices
        
        for i, (mask, class_id) in enumerate(zip(masks, classes)):

            mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
            
            binary_mask = (mask_resized > 0.5).astype(bool)
            
            final_mask[binary_mask] = class_id + 1
    
    return final_mask

def predict_cots_detection(image):
    """Perform COTS detection on input image using YOLO11"""
    if cots_model is None:
        raise Exception("COTS model not loaded")
    
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
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Background stays black (0)
    for class_id, class_info in CORAL_CLASSES.items():
        # Map YOLO class indices (0-7) to mask values (1-8)
        mask_value = class_id + 1
        colored_mask[mask == mask_value] = class_info["color"]
    
    return colored_mask

def create_overlay(image, mask, alpha=0.5):
    colored_mask = create_colored_mask(mask)
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored_mask, alpha, 0)
    else:
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
        "auto_crop_model_loaded": auto_crop_model is not None,
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
        
        # Get optional parameters
        use_auto_crop = request.form.get('use_auto_crop', 'true').lower() == 'true'
        
        # Read and process image - match training preprocessing
        image_bytes = file.read()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        # Convert BGR to RGB (OpenCV uses BGR, but PIL/YOLO expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply auto-crop if enabled and model is loaded
        cropped_images = [image_rgb]
        crop_info = []
        
        if use_auto_crop and auto_crop_model is not None:
            print("Applying auto-crop to image...")
            cropped_images, crop_info = auto_crop_quadrats(image_rgb, CROP_CONF_THRESHOLD)
            print(f"Auto-crop produced {len(cropped_images)} images")
        
        # Process each cropped image
        all_results = []
        
        for idx, cropped_image in enumerate(cropped_images):
            # Perform segmentation on cropped image
            mask = predict_segmentation(cropped_image)
            
            # Create outputs
            colored_mask = create_colored_mask(mask)
            overlay = create_overlay(cropped_image, mask)
            
            # Calculate statistics
            stats = calculate_coral_statistics(mask)
            
            # Convert images to base64
            cropped_b64 = image_to_base64(cropped_image)
            mask_b64 = image_to_base64(colored_mask)
            overlay_b64 = image_to_base64(overlay)
            
            result = {
                "success": True,
                "statistics": stats,
                "images": {
                    "original": f"data:image/png;base64,{cropped_b64}",
                    "mask": f"data:image/png;base64,{mask_b64}",
                    "overlay": f"data:image/png;base64,{overlay_b64}"
                },
                "crop_index": idx,
                "is_cropped": len(cropped_images) > 1
            }
            
            # Add crop info if available
            if idx < len(crop_info):
                result["crop_info"] = crop_info[idx]
            
            all_results.append(result)
        
        # If multiple quadrats were found, include original image with annotations
        original_b64 = image_to_base64(image_rgb)
        
        response = {
            "success": True,
            "results": all_results,
            "total_quadrats": len(cropped_images),
            "auto_crop_applied": use_auto_crop and len(cropped_images) > 1,
            "original_image": f"data:image/png;base64,{original_b64}",
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
        import traceback
        traceback.print_exc()
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
    print(f"🔲 Auto-crop model loaded: {auto_crop_model is not None}")
    print(f"🎯 Device: {DEVICE}")
    print(f"🖼️ Image size: {IMG_SIZE}")
    print(f"🎯 Segmentation confidence threshold: {CONF_THRESHOLD}")
    print(f"⭐ COTS confidence threshold: {COTS_CONF_THRESHOLD}")
    print(f"🔲 Crop confidence threshold: {CROP_CONF_THRESHOLD}")
    print(f"🪸 Number of coral classes: {len(CORAL_CLASSES)}")
    print(f"⭐ Number of COTS classes: {len(COTS_CLASSES)}")
    app.run(host='0.0.0.0', port=port)