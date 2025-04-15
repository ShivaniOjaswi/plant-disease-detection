# PlantIn - Plant Disease Identifier

A web application that identifies plant diseases from images using machine learning.

## Features

- Upload plant images to detect diseases
- Get detailed treatment recommendations
- Browse common plant diseases
- Ask a botanist for personalized advice

## Setup Instructions

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up the model:
   ```
   python mock_train_model.py
   ```
   This creates a mock model specifically trained to detect Tomato Leaf Mold.

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Technical Details

This application uses:
- Flask for the web server
- TensorFlow/Keras for the machine learning model (or a mock model for demonstration)
- Bootstrap for the frontend styling

## Current Status

The application is currently running in mock mode:
- When TensorFlow is not installed, it uses a mock model that simulates disease detection
- The enhanced mock model specifically identifies Tomato Leaf Mold with high confidence
- For real predictions, install TensorFlow using the requirements.txt file

## Note for Production

This is a development version. For production:
- Use a proper WSGI server (like Gunicorn)
- Set up proper authentication
- Configure proper database storage for user uploads
- Set up HTTPS

## License

MIT 