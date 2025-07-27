from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Download model from Google Drive if not already present
MODEL_PATH = "currency_model.h5"
DRIVE_FILE_ID = "1J73rEsSU1q2D1NUi3O3pWgYG5EFb_yso"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load your model
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]
    return "Fake Currency" if prediction >= 0.5 else "Real Currency"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            result = model_predict(filepath)
            image_path = filepath

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

