import os
import requests
import zipfile
import io
import shutil
from tqdm import tqdm

def download_file(url, destination):
    """Download a file from URL with a progress bar"""
    print(f"Downloading from {url}...")
    
    # Make a streaming GET request
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create a progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
    
    # Write the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Download incomplete")
        return False
    
    return True

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # URL for pre-trained plant disease model (replace with actual hosted model)
    model_url = "https://github.com/user/repo/releases/download/v1.0/plant_disease_model.h5"
    
    # Destination path
    model_path = "models/plant_disease_model.h5"
    
    # If model already exists, ask for confirmation to overwrite
    if os.path.exists(model_path):
        response = input(f"Model already exists at {model_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled. Using existing model.")
            return
    
    try:
        # Download the model file
        success = download_file(model_url, model_path)
        
        if success:
            print(f"Model successfully downloaded to {model_path}")
        else:
            print("Failed to download model.")
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("\nAlternatively, you can train your own model using the train_model.py script.")

if __name__ == "__main__":
    main() 