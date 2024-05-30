# Coin Detection API

This project provides a solution for detecting circular objects (coins) in images using a YOLOv5 model and a Flask API. The solution includes endpoints for uploading images, detecting coins, and retrieving details of detected coins.

## Features

- Train a YOLOv5 model to detect coins in images.
- Upload images and store them in persistent storage.
- Detect coins in uploaded images.
- Retrieve list of all detected coins with their IDs and bounding boxes.
- Retrieve details (bounding box, centroid, radius) of specific coins by ID.
- Containerized solution using Docker.

## Setup

### Prerequisites

- Python 3.8
- Docker (if using containerization)

### Install Dependencies

First, clone the YOLOv5 repository and navigate to it:

```sh
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
### Prepare the Dataset
```sh
pip install -r requirements.txt
python convert_data.py
```
It organizes the data in the below format
```commandline
data/
── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
### Train the YOLOv5 Model
Run the following command to train the model:
```sh
python train.py --img 640 --batch 16 --epochs 50 --data yolov5/coin_dataset.yaml --weights yolov5s.pt
```
Parameters
```--img 640: Image size
--batch 16: Batch size
--epochs 50: Number of epochs
--data coin_dataset.yaml: Path to dataset configuration file
--cfg yolov5s.yaml: Model configuration file
--weights yolov5s.pt: Pre-trained weights file
--name yolov5_coins: Name of the training run
```
### Directly build the docker image to run the application

### Build docker Image
```commandline
docker build -t coin-detector .
```

### Run the Docker container
```commandline
docker run -p 5000:5000 coin-detector
```

### Using the Flask API Inside the Docker Container
Upload an image:
```sh
curl -X POST -F 'file=@/path/to/your/image.png' http://localhost:5000/upload
```
Detect coins:
```sh
curl -X POST -H "Content-Type: application/json" -d '{"image_path": "uploads/your_image.png"}' http://localhost:5000/detect
```
Get details of a specific coin:
```sh
curl -X GET http://localhost:5000/coin/<coin_id>
```

### Evalution Metrics
As part of YOLOv5, we are calculating below metrics.

**Precision**: The fraction of true positive detections among all positive detections. It measures how accurate the detections are.

**Recall**: The fraction of true positive detections among all actual positive instances. It measures how well the model detects all relevant objects.

**Mean Average Precision (mAP)**: The average of the precision values at different recall levels. mAP@0.5 (or simply mAP) is calculated at an Intersection over Union (IoU) threshold of 0.5, while mAP@0.5:0.95 is averaged over IoU thresholds from 0.5 to 0.95 in increments of 0.05.

**IoU (Intersection over Union)**: A measure of the overlap between the predicted bounding box and the ground truth bounding box.

The model that is inferences here is having all the evaluation metrics with 98% score. Refer to the results csv file in the exp2 folder in yolov5/runs directory.

