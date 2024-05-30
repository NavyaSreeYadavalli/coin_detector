import os
import json
import shutil

# Define paths
source_image_dir = 'data/coin-dataset'  # Directory with source images
annotation_file = 'data/coin-dataset/_annotations.coco.json'  # Path to COCO JSON annotations
output_dir = 'yolov5/data'  # Output directory

# Create output directories
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

# Load COCO annotations
with open(annotation_file) as f:
    coco_data = json.load(f)

# Extract categories
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
category_ids = {name: idx for idx, name in enumerate(categories.values())}


def convert_to_yolo_format(annotation, img_width, img_height):
    x_center = (annotation['bbox'][0] + annotation['bbox'][2] / 2) / img_width
    y_center = (annotation['bbox'][1] + annotation['bbox'][3] / 2) / img_height
    width = annotation['bbox'][2] / img_width
    height = annotation['bbox'][3] / img_height
    return [annotation['category_id']-1, x_center, y_center, width, height]


# Split data into train and val (80-20 split)
train_split = 0.8
num_images = len(coco_data['images'])
train_size = int(num_images * train_split)

for idx, img_info in enumerate(coco_data['images']):
    img_id = img_info['id']
    img_filename = img_info['file_name']
    img_width, img_height = img_info['width'], img_info['height']

    if idx < train_size:
        image_output_path = os.path.join(output_dir, 'images/train', img_filename)
        label_output_path = os.path.join(output_dir, 'labels/train', os.path.splitext(img_filename)[0] + '.txt')
    else:
        image_output_path = os.path.join(output_dir, 'images/val', img_filename)
        label_output_path = os.path.join(output_dir, 'labels/val', os.path.splitext(img_filename)[0] + '.txt')

    # Copy image to output directory
    shutil.copy(os.path.join(source_image_dir, img_filename), image_output_path)

    # Write label file in YOLO format
    with open(label_output_path, 'w') as f:
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == img_id:
                yolo_annotation = convert_to_yolo_format(annotation, img_width, img_height)
                f.write(' '.join(map(str, yolo_annotation)) + '\n')

print("Dataset conversion complete.")
