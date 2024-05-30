from flask import Flask, request, jsonify, send_file
import torch
import os
from PIL import Image
import uuid

app = Flask(__name__)
detections_store = {}
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'yolov5/runs/train/exp2/weights/best.pt'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

def detect_coins(image_path):
    img = Image.open(image_path)
    results = model(img)
    detections = results.pandas().xyxy[0]
    coins = []
    for _, row in detections.iterrows():
        if row['name'] == 'coin':  # Assuming 'coin' is the class name for coin objects
            coin_id = str(uuid.uuid4())
            coin_data = {
                'id': coin_id,
                'bounding_box': [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])],
                'confidence': float(row['confidence']),
                'centroid': [(int(row['xmin']) + int(row['xmax'])) // 2, (int(row['ymin']) + int(row['ymax'])) // 2],
                'radius': (int(row['xmax']) - int(row['xmin'])) // 2
            }
            coins.append(coin_data)
        detections_store[coin_id] = coin_data
    return coins

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.get_json()
    image_path = data.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400
    coins = detect_coins(image_path)
    return jsonify({'coins': coins}), 200

@app.route('/coin/<string:coin_id>', methods=['GET'])
def get_coin_details(coin_id):
    coin = detections_store.get(coin_id)
    if coin:
        return jsonify(coin), 200
    else:
        return jsonify({'error': 'Coin not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
