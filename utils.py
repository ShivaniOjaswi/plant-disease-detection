import os
import numpy as np
from PIL import Image

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_uploaded_file(uploaded_file, upload_folder="uploads"):
    """Save the uploaded file to a temporary directory"""
    ensure_dir(upload_folder)
    file_path = os.path.join(upload_folder, uploaded_file.filename)
    
    # Save the file
    uploaded_file.save(file_path)
    return file_path

def format_plant_name(disease_class):
    """Format disease class name for display"""
    if disease_class is None:
        return "Unknown"
    
    # Replace underscores with spaces
    parts = disease_class.split('___')
    
    if len(parts) != 2:
        return disease_class.replace('_', ' ')
    
    plant, condition = parts
    plant = plant.replace('_', ' ')
    condition = condition.replace('_', ' ')
    
    if condition.lower() == 'healthy':
        return f"{plant} (Healthy)"
    else:
        return f"{plant} - {condition}"

def decode_predictions(predictions, top=3):
    """Decode model predictions and return top N results"""
    from model import DISEASE_CLASSES
    
    # Get indices of top predictions
    top_indices = predictions[0].argsort()[-top:][::-1]
    
    results = []
    for i in top_indices:
        disease_class = DISEASE_CLASSES.get(i, "Unknown")
        confidence = predictions[0][i] * 100
        results.append({
            'class_index': i,
            'disease_class': disease_class,
            'formatted_name': format_plant_name(disease_class),
            'confidence': confidence
        })
    
    return results 