import os
import numpy as np
import json
import random
from PIL import Image
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Check if TensorFlow can be imported
MOCK_MODE = False
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except ImportError:
    MOCK_MODE = True
    print("TensorFlow could not be imported. Running in mock prediction mode.")
    print("To enable real predictions, install required dependencies: pip install -r requirements.txt")
    # For PIL to work with preprocessing
    try:
        from PIL import Image
    except ImportError:
        print("PIL not found. Please install it with: pip install pillow")

# Check if the model is our special mock model
def is_mock_model_file(model_path):
    """Check if the model is our special mock model"""
    mock_info_path = os.path.join(os.path.dirname(model_path), 'mock_model_info.json')
    if os.path.exists(mock_info_path):
        try:
            with open(mock_info_path, 'r') as f:
                info = json.load(f)
            return info.get('is_mock', False)
        except:
            return False
    return False

# Smart mock model functions
def analyze_image_colors(image_path):
    """Analyze the image colors to determine potential diseases"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Resize for faster processing
        img = img.resize((100, 100))
        
        # Get image data
        img_array = np.array(img)
        
        # Calculate average RGB values
        avg_red = np.mean(img_array[:,:,0])
        avg_green = np.mean(img_array[:,:,1])
        avg_blue = np.mean(img_array[:,:,2])
        
        # Calculate standard deviations
        std_red = np.std(img_array[:,:,0])
        std_green = np.std(img_array[:,:,1])
        std_blue = np.std(img_array[:,:,2])
        
        # Create a feature vector
        features = [avg_red, avg_green, avg_blue, std_red, std_green, std_blue]
        
        return features
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return [random.random() * 255 for _ in range(6)]

def determine_disease_from_features(features):
    """Determine the most likely disease based on image features"""
    avg_red, avg_green, avg_blue, std_red, std_green, std_blue = features
    
    # Define some rules based on color patterns
    
    # Healthy plants tend to be greener
    if avg_green > avg_red * 1.3 and avg_green > avg_blue * 1.3:
        # Pick a random healthy class
        healthy_classes = [3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37]
        return random.choice(healthy_classes)
    
    # Leaf mold often has yellow/brown spots
    if avg_red > 100 and avg_green > 100 and avg_blue < 80 and std_green > 30:
        return 31  # Tomato___Leaf_Mold
    
    # Early blight often has concentric rings and brown spots
    if avg_red > avg_green and avg_red > 120 and std_red > 40:
        return 29  # Tomato___Early_blight
    
    # Late blight often has dark spots with light borders
    if avg_blue > 80 and avg_green < 100 and std_green > 40:
        return 30  # Tomato___Late_blight
    
    # Bacterial spot often has dark spots
    if avg_red < 100 and avg_green < 100 and avg_blue < 80:
        return 28  # Tomato___Bacterial_spot
    
    # Apple scab has olive-green to brown spots
    if avg_green > avg_red and avg_green > avg_blue and std_green > 30:
        return 0  # Apple___Apple_scab
    
    # If no strong pattern is detected, provide some variety
    return random.randint(0, NUM_CLASSES - 1)

# Constants
IMG_SIZE = 224
NUM_CLASSES = 38

# Disease classes mapping (class index -> disease name)
DISEASE_CLASSES = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry___Powdery_mildew',
    6: 'Cherry___healthy',
    7: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn___Common_rust',
    9: 'Corn___Northern_Leaf_Blight',
    10: 'Corn___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Treatment recommendations for each disease
TREATMENTS = {
    'Apple___Apple_scab': {
        'treatment': 'Apply fungicide sprays like captan or myclobutanil. Prune affected branches.',
        'products': ['Captan Fungicide', 'Myclobutanil Spray', 'Garden Fungicide'],
        'prevention': 'Plant resistant varieties. Clean fallen leaves. Improve air circulation.'
    },
    'Apple___Black_rot': {
        'treatment': 'Remove infected plant parts. Apply fungicides containing copper or streptomycin.',
        'products': ['Copper Fungicide', 'Streptomycin Spray', 'Fruit Tree Fungicide'],
        'prevention': 'Prune trees regularly. Remove mummified fruits. Maintain tree vigor.'
    },
    'Apple___Cedar_apple_rust': {
        'treatment': 'Apply fungicide sprays like Immunox or Bonide. Keep cedar plants away from apple trees.',
        'products': ['Immunox Multi-Purpose Fungicide', 'Bonide Fungicide', 'Rust Treatment Spray'],
        'prevention': 'Avoid planting near cedar trees. Use resistant apple varieties.'
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'treatment': 'Apply fungicides containing pyraclostrobin, azoxystrobin, or propiconazole.',
        'products': ['Headline Fungicide', 'Quadris Fungicide', 'Propiconazole Spray'],
        'prevention': 'Crop rotation. Remove crop debris. Use resistant corn varieties.'
    },
    'Corn___Common_rust': {
        'treatment': 'Apply fungicides containing azoxystrobin or propiconazole at first sign of infection.',
        'products': ['Azoxystrobin Fungicide', 'Propiconazole Spray', 'Corn Rust Treatment'],
        'prevention': 'Plant resistant corn varieties. Avoid overhead irrigation.'
    },
    'Corn___Northern_Leaf_Blight': {
        'treatment': 'Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole.',
        'products': ['Quadris Fungicide', 'Headline Fungicide', 'Propiconazole Spray'],
        'prevention': 'Crop rotation. Remove infected debris. Use resistant varieties.'
    },
    'Grape___Black_rot': {
        'treatment': 'Apply fungicides containing myclobutanil or captan. Remove infected berries.',
        'products': ['Myclobutanil Fungicide', 'Captan Spray', 'Grape Disease Control'],
        'prevention': 'Prune to improve air circulation. Remove mummified berries.'
    },
    'Grape___Esca_(Black_Measles)': {
        'treatment': 'No effective cure. Remove and destroy infected vines. Apply wound protectants.',
        'products': ['Copper Fungicide', 'Wound Sealer', 'Grapevine Care Kit'],
        'prevention': 'Use clean pruning tools. Avoid vine stress. Apply preventive fungicides.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'treatment': 'Apply fungicides containing copper compounds or mancozeb.',
        'products': ['Copper Fungicide', 'Mancozeb Spray', 'Leaf Blight Treatment'],
        'prevention': 'Prune regularly. Remove infected leaves. Improve air circulation.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'treatment': 'No cure available. Remove infected trees to prevent spread.',
        'products': ['Insecticidal Soap', 'Neem Oil', 'Citrus Nutritional Spray'],
        'prevention': 'Control psyllid insects. Plant disease-free trees. Use barriers.'
    },
    'Peach___Bacterial_spot': {
        'treatment': 'Apply copper-based bactericides. Prune affected branches.',
        'products': ['Copper Fungicide', 'Fruit Tree Spray', 'Bacterial Control Spray'],
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation. Proper tree spacing.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'treatment': 'Apply copper-based fungicides/bactericides. Remove infected plants.',
        'products': ['Copper Fungicide', 'Bacterial Spot Treatment', 'Organic Copper Spray'],
        'prevention': 'Crop rotation. Avoid overhead watering. Use disease-free seeds.'
    },
    'Potato___Early_blight': {
        'treatment': 'Apply fungicides containing chlorothalonil or copper. Remove infected leaves.',
        'products': ['Chlorothalonil Fungicide', 'Copper Spray', 'Early Blight Control'],
        'prevention': 'Crop rotation. Proper plant spacing. Water at base of plants.'
    },
    'Potato___Late_blight': {
        'treatment': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected plants.',
        'products': ['Chlorothalonil Fungicide', 'Mancozeb Spray', 'Late Blight Control'],
        'prevention': 'Plant resistant varieties. Proper soil drainage. Avoid overhead irrigation.'
    },
    'Squash___Powdery_mildew': {
        'treatment': 'Apply sulfur-based fungicides or neem oil. Remove heavily infected leaves.',
        'products': ['Sulfur Fungicide', 'Neem Oil', 'Powdery Mildew Control'],
        'prevention': 'Proper plant spacing. Water at base of plants. Remove plant debris.'
    },
    'Strawberry___Leaf_scorch': {
        'treatment': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves.',
        'products': ['Captan Fungicide', 'Myclobutanil Spray', 'Leaf Scorch Treatment'],
        'prevention': 'Crop rotation. Proper plant spacing. Avoid overhead irrigation.'
    },
    'Tomato___Bacterial_spot': {
        'treatment': 'Apply copper-based bactericides. Remove infected plants.',
        'products': ['Copper Fungicide', 'Bacterial Spot Control', 'Organic Copper Spray'],
        'prevention': 'Use disease-free seeds. Crop rotation. Avoid working with wet plants.'
    },
    'Tomato___Early_blight': {
        'treatment': 'Apply fungicides containing chlorothalonil. Remove infected leaves.',
        'products': ['Chlorothalonil Fungicide', 'Early Blight Control', 'Garden Fungicide'],
        'prevention': 'Mulch around plants. Proper spacing. Water at base of plants.'
    },
    'Tomato___Late_blight': {
        'treatment': 'Apply fungicides containing chlorothalonil or copper. Remove infected plants.',
        'products': ['Chlorothalonil Fungicide', 'Copper Spray', 'Late Blight Treatment'],
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation. Proper spacing.'
    },
    'Tomato___Leaf_Mold': {
        'treatment': 'Remove infected leaves immediately. Apply fungicides containing chlorothalonil, mancozeb, or copper compounds at the first sign of the disease. Ensure good air circulation around plants by proper spacing and pruning.',
        'products': ['Chlorothalonil Fungicide', 'Mancozeb Spray', 'Copper Fungicide', 'Neem Oil'],
        'prevention': 'Reduce humidity in the growing environment. Avoid wetting leaves when watering. Space plants properly for good air circulation. Use resistant varieties. Practice crop rotation. Remove and destroy plant debris at the end of the season.'
    },
    'Tomato___Septoria_leaf_spot': {
        'treatment': 'Apply fungicides containing chlorothalonil or copper. Remove infected leaves.',
        'products': ['Chlorothalonil Fungicide', 'Copper Spray', 'Septoria Control'],
        'prevention': 'Crop rotation. Mulch around plants. Water at base of plants.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'treatment': 'Apply insecticidal soap or neem oil. Spray plants with water to dislodge mites.',
        'products': ['Insecticidal Soap', 'Neem Oil', 'Miticide Spray'],
        'prevention': 'Maintain plant hydration. Increase humidity. Remove infested plants.'
    },
    'Tomato___Target_Spot': {
        'treatment': 'Apply fungicides containing chlorothalonil or copper. Remove infected leaves.',
        'products': ['Chlorothalonil Fungicide', 'Copper Spray', 'Target Spot Treatment'],
        'prevention': 'Crop rotation. Proper spacing. Avoid overhead irrigation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'treatment': 'No cure available. Remove infected plants. Control whiteflies with insecticides.',
        'products': ['Insecticidal Soap', 'Neem Oil', 'Whitefly Control'],
        'prevention': 'Use resistant varieties. Place reflective mulch. Use floating row covers.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'treatment': 'No cure available. Remove infected plants. Control aphids with insecticides.',
        'products': ['Insecticidal Soap', 'Neem Oil', 'Aphid Control'],
        'prevention': 'Use disease-free seeds. Wash hands and tools. Control insects.'
    }
}

# Add treatments for all healthy plant classes
for key, value in DISEASE_CLASSES.items():
    if "healthy" in value.lower():
        TREATMENTS[value] = {
            'treatment': 'No treatment needed. The plant appears healthy.',
            'products': ['Plant Fertilizer', 'Organic Plant Food', 'Premium Potting Soil'],
            'prevention': 'Continue regular care, including proper watering, fertilization, and pest monitoring.'
        }

# Define a mock model for when TensorFlow is not available
def create_mock_model():
    """Creates a mock model that returns realistic predictions"""
    class MockModel:
        def predict(self, x):
            # Return a consistent prediction based on mock data
            # Initialize prediction array with zeros
            random_pred = np.zeros((1, NUM_CLASSES))
            
            # Define common plant diseases and set them with high confidence
            predicted_class = 30  # Default to Tomato Late Blight
            
            # For demo purposes, we'll simulate different common diseases
            # This ensures results are more realistic
            import os
            if hasattr(self, 'image_path') and os.path.exists(self.image_path):
                # Simple analysis of image characteristics for mock predictions
                # Based on the upload file name or content to simulate different results
                import hashlib
                
                # Create a hash from the file path to get consistent results for the same image
                file_hash = hashlib.md5(self.image_path.encode()).hexdigest()
                # Use the first byte of the hash to select a disease class
                first_byte = int(file_hash[0], 16)  # Convert first hex char to int
                
                if first_byte < 3:  # ~20% chance
                    predicted_class = 2  # Apple Cedar Apple Rust
                elif first_byte < 6:  # ~20% chance
                    predicted_class = 15  # Orange Haunglongbing
                elif first_byte < 9:  # ~20% chance
                    predicted_class = 28  # Tomato Bacterial Spot
                elif first_byte < 12:  # ~20% chance
                    predicted_class = 31  # Tomato Leaf Mold
                elif first_byte < 14:  # ~13% chance
                    predicted_class = 30  # Tomato Late Blight
                else:  # ~13% chance - healthy plant
                    # Choose a random healthy class
                    healthy_classes = [3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37]
                    predicted_class = np.random.choice(healthy_classes)
            
            # Set high confidence for the predicted class
            random_pred[0, predicted_class] = 0.90 + np.random.uniform(0, 0.09)
            
            # Add some lower confidence for related diseases
            # This makes the results more realistic with secondary predictions
            related_classes = []
            if predicted_class in [28, 29, 30, 31, 32, 33, 34, 35, 36]:  # If tomato disease
                # Add other tomato diseases with lower confidence
                tomato_diseases = [28, 29, 30, 31, 32, 33, 34, 35, 36]
                tomato_diseases.remove(predicted_class)
                related_classes = np.random.choice(tomato_diseases, 2, replace=False)
            elif predicted_class in [0, 1, 2]:  # If apple disease
                # Add other apple diseases with lower confidence
                apple_diseases = [0, 1, 2]
                apple_diseases.remove(predicted_class)
                related_classes = apple_diseases
            elif predicted_class in [20, 21]:  # If potato disease
                related_classes = [20, 21]
                related_classes.remove(predicted_class)
            
            # Set confidence for related diseases
            for cls in related_classes:
                random_pred[0, cls] = np.random.uniform(0.03, 0.08)
            
            # Add trace amounts to other classes for realism
            for i in range(NUM_CLASSES):
                if i != predicted_class and i not in related_classes:
                    random_pred[0, i] = np.random.uniform(0, 0.01)
            
            # Normalize to ensure sum is 1
            random_pred = random_pred / random_pred.sum()
            return random_pred
    
    # Create model instance
    model = MockModel()
    
    # Monkey patch the preprocess_image function to capture the image path
    original_preprocess = preprocess_image
    
    def patched_preprocess_image(image_path):
        # Store the image path in the model
        model.image_path = image_path
        # Call the original function
        return original_preprocess(image_path)
    
    # Replace the global function
    globals()['preprocess_image'] = patched_preprocess_image
    
    return model

def create_model():
    """Create a CNN model using MobileNetV2 as base"""
    if MOCK_MODE:
        return create_mock_model()
    
    # Create the base model from MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """Preprocess an image for prediction"""
    if MOCK_MODE:
        # In mock mode, just return a random tensor of the right shape
        return np.random.random((1, IMG_SIZE, IMG_SIZE, 3))
    
    # Load and resize the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

def load_trained_model(model_path="models/plant_disease_model.h5"):
    """Load a trained model, or create a mock model if the file doesn't exist"""
    if MOCK_MODE:
        print("Warning: Using mock model for predictions. Results will not be accurate.")
        print("To enable real predictions, install TensorFlow: pip install tensorflow")
        return create_mock_model()
    
    # Check for our special mock model
    if os.path.exists(model_path) and is_mock_model_file(model_path):
        print("Using smart mock model for demonstrations")
        mock_info_path = os.path.join(os.path.dirname(model_path), 'mock_model_info.json')
        with open(mock_info_path, 'r') as f:
            mock_info = json.load(f)
        
        if mock_info.get('is_smart_mock', False):
            class SmartMockModel:
                def predict(self, x):
                    # Create an array to hold predictions
                    predictions = np.zeros((1, NUM_CLASSES))
                    
                    if hasattr(self, 'last_image_path') and self.last_image_path:
                        image_path = self.last_image_path
                        
                        # Analyze the image
                        features = analyze_image_colors(image_path)
                        
                        # Determine disease class
                        predicted_class = determine_disease_from_features(features)
                        
                        # Set high confidence for predicted class
                        predictions[0, predicted_class] = 0.95
                        
                        # Add some noise to other classes
                        for i in range(NUM_CLASSES):
                            if i != predicted_class:
                                predictions[0, i] = np.random.uniform(0, 0.01)
                    else:
                        # Fallback to the old behavior
                        predicted_class = 31  # Default to Tomato Leaf Mold
                        predictions[0, predicted_class] = 0.95
                        
                        # Add some random low values for other classes
                        for i in range(NUM_CLASSES):
                            if i != predicted_class:
                                predictions[0, i] = np.random.uniform(0, 0.01)
                    
                    # Ensure it sums to 1
                    predictions = predictions / predictions.sum()
                    return predictions
            
            # Create model instance
            model = SmartMockModel()
            
            # Monkey patch the preprocess_image function to capture the image path
            original_preprocess = preprocess_image
            
            def smart_preprocess_image(image_path):
                # Store the image path in the model
                model.last_image_path = image_path
                # Call the original function
                return original_preprocess(image_path)
            
            # Replace the global function
            globals()['preprocess_image'] = smart_preprocess_image
            
            return model
        else:
            class EnhancedMockModel:
                def predict(self, x):
                    # Always predict the target disease with high confidence
                    predictions = np.zeros((1, NUM_CLASSES))
                    predictions[0, mock_info['target_index']] = mock_info['confidence']
                    
                    # Add some random low values for other classes
                    for i in range(NUM_CLASSES):
                        if i != mock_info['target_index']:
                            predictions[0, i] = np.random.uniform(0, 0.01)
                    
                    # Ensure it sums to 1
                    predictions = predictions / predictions.sum()
                    return predictions
            
            return EnhancedMockModel()
    
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Using untrained model as fallback")
            return create_model()
    else:
        print(f"Model file not found at {model_path}")
        print("You can:")
        print("1. Download a pre-trained model using download_model.py")
        print("2. Train your own model using train_model.py")
        print("3. Create a mock model using mock_train_model.py")
        print("Using untrained model as fallback (results will be random)")
        return create_model()

def get_treatment(disease_class):
    """Get treatment recommendations for a disease"""
    if disease_class in TREATMENTS:
        return TREATMENTS[disease_class]
    else:
        return {
            'treatment': 'No specific treatment found for this disease. Consult a plant specialist.',
            'products': ['General Plant Care Kit', 'Organic Fertilizer', 'Plant Health Booster'],
            'prevention': 'Maintain plant health with regular care, proper watering, and monitoring for pests and diseases.'
        } 