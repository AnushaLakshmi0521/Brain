from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('brain.h5')

# Class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Path to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_and_predict(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))  # Resize the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

    # Make predictions
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user doesn't select a file, the browser may submit an empty part
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Secure and save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the class of the uploaded image
            predicted_class = load_and_predict(file_path)

            # Provide information about the predicted tumor class
            tumor_info = ''
            if predicted_class == 'glioma_tumor':
                tumor_info = ("Glioma is a growth of cells that starts in the brain or spinal cord. "
                              "Symptoms include headache, nausea, confusion, etc.")
            elif predicted_class == 'meningioma_tumor':
                tumor_info = ("Meningioma is a tumor that grows from the membranes that surround the brain. "
                              "Symptoms include changes in vision, headaches, hearing loss, etc.")
            elif predicted_class == 'no_tumor':
                tumor_info = "No tumor detected."
            elif predicted_class == 'pituitary_tumor':
                tumor_info = ("Pituitary tumors are unusual growths that develop in the pituitary gland. "
                              "Symptoms include headaches, eye problems, pain in the face, etc.")

            return render_template('result.html', predicted_class=predicted_class, tumor_info=tumor_info, image_path=file_path)

    return render_template('index.html')

# Route to display the result page
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)
