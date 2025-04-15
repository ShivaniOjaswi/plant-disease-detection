import os
import numpy as np
import json
import h5py
from PIL import Image
import hashlib
import random

# Import disease classes from model.py
from model import DISEASE_CLASSES, NUM_CLASSES

# Set random seed for reproducibility
random.seed(42)

def compute_image_hash(image_path):
    """Compute a hash of the image to use as a consistent identifier"""
    try:
        with open(image_path, 'rb') as f:
            # Use only the first 10KB of the file for speed
            return hashlib.md5(f.read(10240)).hexdigest()
    except Exception as e:
        print(f"Error computing hash: {e}")
        # Return a random hash if there's an error
        return hashlib.md5(str(random.random()).encode()).hexdigest()

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
    
    # If no strong pattern is detected, default to Tomato Leaf Mold for demonstration
    return 31  # Tomato___Leaf_Mold

def create_smart_mock_model():
    """
    Create a mock model file that can produce different results for different images
    """
    print("Creating a smart mock model for plant disease detection...")
    print("This model will analyze image characteristics to make predictions.")
    
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Path for the model
    model_path = 'models/plant_disease_model.h5'
    
    # Check if model already exists
    if os.path.exists(model_path):
        response = input(f"Model already exists at {model_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    try:
        # Create a simple HDF5 file that mimics a model file structure
        with h5py.File(model_path, 'w') as f:
            # Create a group that would typically store model weights
            model_weights = f.create_group('model_weights')
            
            # Create a dataset with some random values
            model_weights.create_dataset('layer_1', data=np.random.random((10, 10)))
            
            # Store configuration
            mock_data = {
                'is_smart_mock': True,
                'version': '1.0'
            }
            
            # Store as JSON string in a dataset
            mock_config = json.dumps(mock_data)
            f.create_dataset('mock_config', data=np.array([mock_config], dtype=h5py.special_dtype(vlen=str)))
        
        print(f"Smart mock model created successfully at {model_path}")
        
        # Create a companion file with mock configuration
        with open('models/mock_model_info.json', 'w') as f:
            json.dump({
                'is_mock': True,
                'is_smart_mock': True,
                'version': '1.0',
                'supported_diseases': list(DISEASE_CLASSES.values())
            }, f, indent=2)
        
        # Create a class mapping file
        with open('models/class_mapping.json', 'w') as f:
            json.dump(DISEASE_CLASSES, f, indent=2)
            
        print("Smart mock model will analyze image colors to determine diseases.")
        print("Different images should now get different predictions.")
            
    except Exception as e:
        print(f"Error creating smart mock model: {str(e)}")

def update_model_py():
    """
    Create instructions for updating model.py
    """
    code_snippet = """
# Add this to model.py

# For smart mock model
def analyze_image_colors(image_path):
    \"\"\"Analyze the image colors to determine potential diseases\"\"\"
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
    \"\"\"Determine the most likely disease based on image features\"\"\"
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

# Then modify the load_trained_model function to include this part:
if os.path.exists(model_path) and is_mock_model_file(model_path):
    print("Using smart mock model for demonstrations")
    mock_info_path = os.path.join(os.path.dirname(model_path), 'mock_model_info.json')
    with open(mock_info_path, 'r') as f:
        mock_info = json.load(f)
        
    if mock_info.get('is_smart_mock', False):
        class SmartMockModel:
            def predict(self, x):
                # Global variable to store the most recent image path
                # Usually would be the last element in the preprocess_image call stack
                global _last_preprocessed_image_path
                
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
"""
    
    print("\nTo enable smart mock predictions, add the following code to model.py:")
    print(code_snippet)
    
    # Also update the actual model.py file
    model_py_path = 'model.py'
    if os.path.exists(model_py_path):
        print("\nWould you like me to automatically update model.py? (y/n): ")
        response = input()
        if response.lower() == 'y':
            try:
                with open(model_py_path, 'r') as f:
                    content = f.read()
                    
                if 'def analyze_image_colors(' not in content:
                    # Update the imports
                    updated_content = content.replace(
                        'import warnings', 
                        'import warnings\nimport random'
                    )
                    
                    # Find the position to add the new functions - after the get_treatment function
                    if 'def get_treatment(' in updated_content:
                        parts = updated_content.split('def get_treatment(')
                        # Find the end of the function
                        second_part = parts[1]
                        function_end = second_part.find('\n\n') + 2
                        if function_end < 2:  # Not found
                            function_end = len(second_part)
                        
                        # Insert our new functions after get_treatment
                        updated_content = (
                            parts[0] + 
                            'def get_treatment(' + 
                            second_part[:function_end] + 
                            '\n\n# Smart mock model functions\n' +
                            'def analyze_image_colors(image_path):\n'
                            '    """Analyze the image colors to determine potential diseases"""\n'
                            '    try:\n'
                            '        img = Image.open(image_path).convert(\'RGB\')\n'
                            '        \n'
                            '        # Resize for faster processing\n'
                            '        img = img.resize((100, 100))\n'
                            '        \n'
                            '        # Get image data\n'
                            '        img_array = np.array(img)\n'
                            '        \n'
                            '        # Calculate average RGB values\n'
                            '        avg_red = np.mean(img_array[:,:,0])\n'
                            '        avg_green = np.mean(img_array[:,:,1])\n'
                            '        avg_blue = np.mean(img_array[:,:,2])\n'
                            '        \n'
                            '        # Calculate standard deviations\n'
                            '        std_red = np.std(img_array[:,:,0])\n'
                            '        std_green = np.std(img_array[:,:,1])\n'
                            '        std_blue = np.std(img_array[:,:,2])\n'
                            '        \n'
                            '        # Create a feature vector\n'
                            '        features = [avg_red, avg_green, avg_blue, std_red, std_green, std_blue]\n'
                            '        \n'
                            '        return features\n'
                            '    except Exception as e:\n'
                            '        print(f"Error analyzing image: {e}")\n'
                            '        return [random.random() * 255 for _ in range(6)]\n'
                            '\n'
                            'def determine_disease_from_features(features):\n'
                            '    """Determine the most likely disease based on image features"""\n'
                            '    avg_red, avg_green, avg_blue, std_red, std_green, std_blue = features\n'
                            '    \n'
                            '    # Define some rules based on color patterns\n'
                            '    \n'
                            '    # Healthy plants tend to be greener\n'
                            '    if avg_green > avg_red * 1.3 and avg_green > avg_blue * 1.3:\n'
                            '        # Pick a random healthy class\n'
                            '        healthy_classes = [3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37]\n'
                            '        return random.choice(healthy_classes)\n'
                            '    \n'
                            '    # Leaf mold often has yellow/brown spots\n'
                            '    if avg_red > 100 and avg_green > 100 and avg_blue < 80 and std_green > 30:\n'
                            '        return 31  # Tomato___Leaf_Mold\n'
                            '    \n'
                            '    # Early blight often has concentric rings and brown spots\n'
                            '    if avg_red > avg_green and avg_red > 120 and std_red > 40:\n'
                            '        return 29  # Tomato___Early_blight\n'
                            '    \n'
                            '    # Late blight often has dark spots with light borders\n'
                            '    if avg_blue > 80 and avg_green < 100 and std_green > 40:\n'
                            '        return 30  # Tomato___Late_blight\n'
                            '    \n'
                            '    # Bacterial spot often has dark spots\n'
                            '    if avg_red < 100 and avg_green < 100 and avg_blue < 80:\n'
                            '        return 28  # Tomato___Bacterial_spot\n'
                            '    \n'
                            '    # Apple scab has olive-green to brown spots\n'
                            '    if avg_green > avg_red and avg_green > avg_blue and std_green > 30:\n'
                            '        return 0  # Apple___Apple_scab\n'
                            '    \n'
                            '    # If no strong pattern is detected, provide some variety\n'
                            '    return random.randint(0, NUM_CLASSES - 1)\n'
                            '\n' +
                            second_part[function_end:]
                        )
                        
                        # Now update the load_trained_model function to handle smart mock models
                        if 'def load_trained_model(' in updated_content:
                            parts = updated_content.split('# Check for our special mock model')
                            if len(parts) > 1:
                                # Replace the smart mock model detection code
                                detection_part = parts[1].split('if os.path.exists(model_path):')
                                if len(detection_part) > 1:
                                    updated_content = (
                                        parts[0] + 
                                        '# Check for our special mock model\n'
                                        '    if os.path.exists(model_path) and is_mock_model_file(model_path):\n'
                                        '        print("Using smart mock model for demonstrations")\n'
                                        '        mock_info_path = os.path.join(os.path.dirname(model_path), \'mock_model_info.json\')\n'
                                        '        with open(mock_info_path, \'r\') as f:\n'
                                        '            mock_info = json.load(f)\n'
                                        '        \n'
                                        '        if mock_info.get(\'is_smart_mock\', False):\n'
                                        '            class SmartMockModel:\n'
                                        '                def predict(self, x):\n'
                                        '                    # Create an array to hold predictions\n'
                                        '                    predictions = np.zeros((1, NUM_CLASSES))\n'
                                        '                    \n'
                                        '                    if hasattr(self, \'last_image_path\') and self.last_image_path:\n'
                                        '                        image_path = self.last_image_path\n'
                                        '                        \n'
                                        '                        # Analyze the image\n'
                                        '                        features = analyze_image_colors(image_path)\n'
                                        '                        \n'
                                        '                        # Determine disease class\n'
                                        '                        predicted_class = determine_disease_from_features(features)\n'
                                        '                        \n'
                                        '                        # Set high confidence for predicted class\n'
                                        '                        predictions[0, predicted_class] = 0.95\n'
                                        '                        \n'
                                        '                        # Add some noise to other classes\n'
                                        '                        for i in range(NUM_CLASSES):\n'
                                        '                            if i != predicted_class:\n'
                                        '                                predictions[0, i] = np.random.uniform(0, 0.01)\n'
                                        '                    else:\n'
                                        '                        # Fallback to the old behavior\n'
                                        '                        predicted_class = 31  # Default to Tomato Leaf Mold\n'
                                        '                        predictions[0, predicted_class] = 0.95\n'
                                        '                        \n'
                                        '                        # Add some random low values for other classes\n'
                                        '                        for i in range(NUM_CLASSES):\n'
                                        '                            if i != predicted_class:\n'
                                        '                                predictions[0, i] = np.random.uniform(0, 0.01)\n'
                                        '                    \n'
                                        '                    # Ensure it sums to 1\n'
                                        '                    predictions = predictions / predictions.sum()\n'
                                        '                    return predictions\n'
                                        '            \n'
                                        '            # Create model instance\n'
                                        '            model = SmartMockModel()\n'
                                        '            \n'
                                        '            # Monkey patch the preprocess_image function to capture the image path\n'
                                        '            original_preprocess = preprocess_image\n'
                                        '            \n'
                                        '            def smart_preprocess_image(image_path):\n'
                                        '                # Store the image path in the model\n'
                                        '                model.last_image_path = image_path\n'
                                        '                # Call the original function\n'
                                        '                return original_preprocess(image_path)\n'
                                        '            \n'
                                        '            # Replace the global function\n'
                                        '            globals()[\'preprocess_image\'] = smart_preprocess_image\n'
                                        '            \n'
                                        '            return model\n'
                                        '        else:\n'
                                        '            class EnhancedMockModel:\n'
                                        '                def predict(self, x):\n'
                                        '                    # Always predict the target disease with high confidence\n'
                                        '                    predictions = np.zeros((1, NUM_CLASSES))\n'
                                        '                    predictions[0, mock_info[\'target_index\']] = mock_info[\'confidence\']\n'
                                        '                    \n'
                                        '                    # Add some random low values for other classes\n'
                                        '                    for i in range(NUM_CLASSES):\n'
                                        '                        if i != mock_info[\'target_index\']:\n'
                                        '                            predictions[0, i] = np.random.uniform(0, 0.01)\n'
                                        '                    \n'
                                        '                    # Ensure it sums to 1\n'
                                        '                    predictions = predictions / predictions.sum()\n'
                                        '                    return predictions\n'
                                        '            \n'
                                        '            return EnhancedMockModel()\n' +
                                        '    if os.path.exists(model_path):' +
                                        detection_part[1]
                                    )
                        
                        with open(model_py_path, 'w') as f:
                            f.write(updated_content)
                            
                        print("Successfully updated model.py with smart mock model functionality.")
                        
            except Exception as e:
                print(f"Error updating model.py: {str(e)}")
                print("Please manually update the file using the code snippet above.")
        else:
            print("Please manually add the code to model.py if you want to use smart mock predictions.")
    else:
        print("model.py file not found. Please manually add the code if you want to use smart mock predictions.")

if __name__ == "__main__":
    try:
        import h5py
    except ImportError:
        print("Error: h5py package is required but not installed.")
        print("Please install it using: pip install h5py")
        exit(1)
        
    create_smart_mock_model()
    update_model_py()
    
    print("\nNext steps:")
    print("1. Run your application: python app.py")
    print("2. Upload different plant images")
    print("3. The system will analyze each image and provide different predictions based on image characteristics") 