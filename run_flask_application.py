import os
import subprocess
import sys
import webbrowser
from threading import Timer

# Function to install necessary libraries
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure necessary libraries are installed
def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        install(package)

check_and_install("flask")
check_and_install("tensorflow")
check_and_install("pillow")

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np

app = Flask(__name__)

# Ensure necessary directories
if not os.path.exists('flask_app/static'):
    os.makedirs('flask_app/static')

# Load the HDF5 model
model_path = os.path.join(os.getcwd(), 'chest_xray_model2.h5')

print("Model path:", model_path)

# Use load_model to load the HDF5 model
model = load_model(model_path)

# Define the target image size
target_size = (224, 224)
target_conditions = ['Pneumonia', 'Pneumothorax', 'Atelectasis']

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('flask_app/static', file.filename)
            file.save(filepath)

            # Preprocess the image
            img_array = preprocess_image(filepath, target_size)

            # Make prediction
            prediction = model.predict(img_array)

            # Render results
            results = {condition: float(prediction[0][i] * 100) for i, condition in enumerate(target_conditions)}
            return render_template('result.html', results=results, image_path=file.filename)

    return render_template('index.html')

@app.route('/static/<filename>')
def send_file(filename):
    return send_from_directory('flask_app/static', filename)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
