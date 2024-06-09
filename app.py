from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

app = Flask(__name__)

# Cargar el modelo YOLOv4
model = attempt_load('path/to/yolov4-weights.pt', map_location='cpu')

def detect_fruits(image):
    img = letterbox(image, new_shape=640)[0]
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to('cpu').float() / 255.0
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)
    return pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = detect_fruits(image)
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)