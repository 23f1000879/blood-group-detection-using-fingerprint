from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os

app = Flask(__name__)
model = load_model('model_blood_group_detection.h5')

labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)
    predicted_class = np.argmax(result)
    predicted_label = labels[predicted_class]
    confidence = result[0][predicted_class] * 100

    return predicted_label, confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'thumbprint' not in request.files:
            return redirect(request.url)
        file = request.files['thumbprint']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            predicted_label, confidence = model_predict(file_path)
            return render_template('result.html', result=predicted_label, confidence=confidence, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
