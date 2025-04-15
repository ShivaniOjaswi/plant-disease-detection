import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Import our model and utility functions
from model import preprocess_image, load_trained_model, DISEASE_CLASSES, get_treatment
from utils import save_uploaded_file, format_plant_name

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model reference
MODEL = None

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load or create the model on first request"""
    global MODEL
    if MODEL is None:
        from model import MOCK_MODE
        MODEL = load_trained_model()
        # Store mock mode reason in request global
        if MOCK_MODE:
            app.config['MOCK_MODE'] = True
            app.config['MOCK_MODE_REASON'] = "tensorflow not installed"
    return MODEL

def get_all_diseases():
    """Get a list of all diseases"""
    diseases = []
    for idx, disease in DISEASE_CLASSES.items():
        diseases.append({
            'id': idx,
            'name': disease,
            'disease_class': disease,
            'class_index': idx,
            'formatted_name': format_plant_name(disease),
            'treatments': get_treatment(disease)
        })
    return diseases

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        
        # Load the model if not already loaded
        model = load_model()
        
        # Preprocess image for prediction
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get top predictions
        top_indices = predictions[0].argsort()[-3:][::-1]
        
        # Prepare results
        results = []
        for i in top_indices:
            disease_class = DISEASE_CLASSES.get(i, "Unknown")
            confidence = predictions[0][i] * 100
            treatment_info = get_treatment(disease_class)
            
            results.append({
                'class_index': i,
                'disease_class': disease_class,
                'formatted_name': format_plant_name(disease_class),
                'confidence': confidence,
                'treatment': treatment_info
            })
        
        # Create a custom request object with mock mode info
        request.__dict__['mock_mode'] = app.config.get('MOCK_MODE', False)
        request.__dict__['mock_mode_reason'] = app.config.get('MOCK_MODE_REASON', '')
        
        # Return the template with results
        return render_template('result.html', 
                              results=results, 
                              image_file=filename)
    
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        
        # Load the model if not already loaded
        model = load_model()
        
        # Preprocess image for prediction
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get top predictions
        top_indices = predictions[0].argsort()[-3:][::-1]
        
        # Prepare results
        results = []
        for i in top_indices:
            disease_class = DISEASE_CLASSES.get(i, "Unknown")
            confidence = float(predictions[0][i] * 100)
            treatment_info = get_treatment(disease_class)
            
            results.append({
                'class_index': int(i),
                'disease_class': disease_class,
                'formatted_name': format_plant_name(disease_class),
                'confidence': confidence,
                'treatment': treatment_info
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'file': file.filename,
            'predictions': results
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/disease/<int:disease_id>')
def disease_info(disease_id):
    """Show information about a specific disease"""
    if disease_id not in DISEASE_CLASSES:
        return redirect(url_for('index'))
    
    disease_class = DISEASE_CLASSES[disease_id]
    formatted_name = format_plant_name(disease_class)
    treatment_info = get_treatment(disease_class)
    
    return render_template('disease.html', 
                          disease_id=disease_id,
                          disease_class=disease_class,
                          formatted_name=formatted_name,
                          treatment=treatment_info)

@app.route('/ask-botanist')
def ask_botanist():
    """Page for asking a botanist"""
    return render_template('ask_botanist.html')

@app.route('/botanist_thanks', methods=['POST'])
def botanist_thanks():
    """Handle botanist question submission"""
    # In a real app, we would save the form data to a database
    # For demo purposes, we just redirect to a thank you page
    name = request.form.get('name', 'Anonymous')
    return render_template('botanist_thanks.html', name=name)

@app.route('/diseases')
def disease_list():
    """Show list of all diseases"""
    diseases = get_all_diseases()
    return render_template('diseases.html', diseases=diseases)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('privacy.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure model is loaded at startup
    load_model()
    
    # Run Flask app
    app.run(debug=True) 