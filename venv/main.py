from flask import Flask, render_template, request, send_from_directory
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Create a flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']

# Define the Upload folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor Detected: {class_labels[predicted_class_index]}", confidence_score

# Route for the home page
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            # Save the uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the result
            result, confidence = predict_tumor(file_location)
            return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
    

