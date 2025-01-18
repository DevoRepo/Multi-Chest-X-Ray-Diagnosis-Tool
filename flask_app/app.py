from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load the HDF5 model
model_path = 'C:\\Users\\ukkvc\\Downloads\\chest_xray_model.h5'
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
            filepath = os.path.join('static', file.filename)
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
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
