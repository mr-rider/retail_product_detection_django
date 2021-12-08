# Installing YOLOv5 on local machine
# !git clone https://github.com/ultralytics/yolov5  # clone repo
# !pip install -qr yolov5/requirements.txt  # install dependencies (ignore errors)
# %cd yolov5

import torch
from IPython.display import Image, clear_output  # to display images
import cv2
import pandas as pd
import requests
from PIL import Image
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import json
# from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# path to pretrainded model
model_path = '/home/mr-rider/PycharmProjects/retail_product_detection/model/best.pt'
image_path = '/home/mr-rider/PycharmProjects/retail_product_detection/model/2.jpg'
imgs = Image.open(image_path)

model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)

results = model(imgs)
results_json = results.pandas().xyxy[0].to_json(orient="records")
results_json = json.loads(results_json)
#print(results_json)

window_name = 'object_detection'
image_cv = cv2.imread(image_path)

confidence = 0.5
number_of_objects = 0
for detected_object in results_json:
    if detected_object["confidence"] >= 0.5:
        # adding rectangles of objects to input image
        image_with_label = cv2.rectangle(image_cv, (int(detected_object["xmin"]), int(detected_object["ymin"])),
                                         (int(detected_object["xmax"]), int(detected_object["ymax"])), (0, 255, 0), 1)
        number_of_objects += 1

font = cv2.FONT_HERSHEY_COMPLEX
bottom_left_corner_text = (10, 30)
font_scale = 0.5
font_color = (0, 255, 255)
thickness = 1
line_type = 2
img_text = f'Number of objects: {number_of_objects}'


print(f'Number of detected objects: {number_of_objects}')
cv2.putText(image_with_label,
            img_text,
            bottom_left_corner_text,
            font,
            font_scale,
            font_color,
            thickness,
            line_type)
# display image
cv2.imshow(window_name, image_with_label)

# write image
path_to_save_image = '/home/mr-rider/PycharmProjects/retail_product_detection/model/2_label.jpg'
# TODO image_name + labels for saving image
cv2.imwrite(path_to_save_image, image_with_label)
cv2.waitKey()