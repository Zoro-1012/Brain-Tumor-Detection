import os
from flask import Flask, render_template, request, send_from_directory
# Ensure TF_USE_LEGACY_KERAS is set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf # Import tensorflow explicitly
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize Flask app
app = Flask(__name__)

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for model and class labels
model = None
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma'] # Ensure this matches your model's output

# Load the trained model at application startup
# It's crucial that the environment (TensorFlow/Keras version) is compatible
# with how the model was saved. The error 'Unrecognized keyword arguments: ['batch_shape']'
# suggests a version mismatch.
try:
    model_path = 'models/model1.keras'
    if os.path.exists(model_path):
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
    else:
        logging.error(f"Model file not found at {model_path}. Please ensure the model is in the 'models' directory.")
except Exception as e:
    logging.critical(f"Error loading model: {e}. This often indicates a TensorFlow/Keras version incompatibility.")
    logging.critical("Please ensure the TensorFlow/Keras version used to load the model is compatible with the version used to save it.")
    logging.critical("You might need to reinstall specific versions of tensorflow or keras.")
    # Exit or disable prediction features if model fails to load
    model = None

# Helper function to predict tumor type
def predict_tumor(image_path):
    if model is None:
        return "Model not loaded. Cannot perform prediction.", 0.0

    IMAGE_SIZE = 128
    try:
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        predicted_label = class_labels[predicted_class_index]

        if predicted_label == 'notumor':
            return "No Tumor Detected", confidence_score
        else:
            # Capitalize the first letter of the tumor type for better display
            return f"Tumor Type: {predicted_label.capitalize()}", confidence_score
    except Exception as e:
        logging.error(f"Error during prediction for {image_path}: {e}")
        return "Prediction failed due to an error.", 0.0

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    confidence_score = None
    uploaded_file_path = None
    error_message = None

    if request.method == 'POST':
        # Check if a file was provided in the request
        if 'file' not in request.files:
            error_message = "No file part in the request."
        else:
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                error_message = "No selected file."
            else:
                try:
                    # Securely save the file
                    filename = file.filename
                    file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_location)
                    logging.info(f"File saved to: {file_location}")

                    # Predict the tumor
                    prediction_result, confidence = predict_tumor(file_location)
                    confidence_score = f"{confidence*100:.2f}%"
                    uploaded_file_path = f'/uploads/{filename}'

                except Exception as e:
                    logging.error(f"Error processing uploaded file: {e}")
                    error_message = f"An error occurred during file processing: {e}"

    return render_template('index.html',
                           result=prediction_result,
                           confidence=confidence_score,
                           file_path=uploaded_file_path,
                           error=error_message)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Make sure to create a 'models' directory in the same location as main.py
    # and place 'model_legacy.h5' inside it.
    # Also, ensure 'uploads' directory exists or is created by the app.
    app.run(debug=True)
