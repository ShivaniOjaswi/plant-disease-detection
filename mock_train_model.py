import os
import numpy as np
import json
import h5py
from tqdm import tqdm

def create_mock_tomato_leaf_mold_model():
    """
    Create a mock model file specifically for Tomato Leaf Mold detection.
    This doesn't actually create a functioning TensorFlow model, but rather
    a file that looks like a model to the system and will make consistent predictions.
    """
    print("Creating a mock model for Tomato Leaf Mold detection...")
    print("Note: This is not a real trained model but will produce consistent predictions.")
    
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
            
            # Create dataset to store our special prediction behavior
            # This isn't how real models work, but we'll use it to store our mock prediction logic
            mock_data = {
                'target_disease': 'Tomato___Leaf_Mold',
                'target_index': 31,  # Index for Tomato Leaf Mold
                'confidence_level': 0.99  # 99% confidence for our mock detection
            }
            
            # Store as JSON string in a dataset
            mock_config = json.dumps(mock_data)
            f.create_dataset('mock_config', data=np.array([mock_config], dtype=h5py.special_dtype(vlen=str)))
        
        print(f"Mock model created successfully at {model_path}")
        print("This model will consistently predict Tomato Leaf Mold with high confidence.")
        print("Note: This is for demonstration purposes only and does not perform real image analysis.")
        
        # Create a companion file to help the app recognize this as a mock model
        with open('models/mock_model_info.json', 'w') as f:
            json.dump({
                'is_mock': True,
                'target_disease': 'Tomato___Leaf_Mold',
                'target_index': 31,
                'confidence': 0.99
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error creating mock model: {str(e)}")

def create_prediction_patch():
    """
    Create a patch for the model.py file to handle our mock model
    """
    patch_code = """
# Add this near the top of model.py
def is_mock_model_file(model_path):
    \"\"\"Check if the model is our special mock model\"\"\"
    mock_info_path = os.path.join(os.path.dirname(model_path), 'mock_model_info.json')
    if os.path.exists(mock_info_path):
        try:
            with open(mock_info_path, 'r') as f:
                info = json.load(f)
            return info.get('is_mock', False)
        except:
            return False
    return False
    
# Then modify the load_trained_model function to check for mock model
def load_trained_model(model_path="models/plant_disease_model.h5"):
    \"\"\"Load a trained model, or create a mock model if the file doesn't exist\"\"\"
    if MOCK_MODE:
        print("Warning: Using mock model for predictions. Results will not be accurate.")
        print("To enable real predictions, install TensorFlow: pip install tensorflow")
        return create_mock_model()
    
    # Check for our special mock model
    if os.path.exists(model_path) and is_mock_model_file(model_path):
        print("Using enhanced mock model for demonstrations")
        mock_info_path = os.path.join(os.path.dirname(model_path), 'mock_model_info.json')
        with open(mock_info_path, 'r') as f:
            mock_info = json.load(f)
            
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
"""
    
    print("\nTo enhance the mock model functionality, consider adding the following code to model.py:")
    print(patch_code)
    print("\nThis is optional, as your current code will work with the mock model we created.")

if __name__ == "__main__":
    try:
        import h5py
    except ImportError:
        print("Error: h5py package is required but not installed.")
        print("Please install it using: pip install h5py")
        exit(1)
        
    create_mock_tomato_leaf_mold_model()
    create_prediction_patch()
    
    print("\nNext steps:")
    print("1. Run your application: python app.py")
    print("2. Upload any plant image")
    print("3. The system will identify it as Tomato Leaf Mold with high confidence") 