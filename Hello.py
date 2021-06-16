import json
import numpy as np
import torch
#from model_eff import load_model
from melanoma_service.model_eff import load_model
#from predict import predict
from time import time
from efficientnet_pytorch import EfficientNet
import torchtoolbox.transform as transforms
import torch.nn as nn
import torch
import cv2
import os
    
  
def data_format(data):
    
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    #print("jani",data[0][0])
    script_dir = os.path.dirname(__file__)
    print(script_dir)
    img_path = os.path.join(script_dir,".\Data\\"+data[0][0])
    print("jani",img_path)
    x = cv2.imread(img_path)
    x = test_transform(x).unsqueeze(0)
    meta = torch.tensor(data[1]).unsqueeze(0)
    meta = meta.type(torch.FloatTensor)

    return (x,meta)

# Called when a request is received
def run(raw_data):

    data = data_format(json.loads(raw_data)['body']['keys'])
    # Get a prediction from the model
    z_val = model(data)
    predictions = torch.sigmoid(z_val)
    predictions =  predictions.type(torch.int).item()
    print(predictions)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['Not-Melanoma', 'Melanoma']
    predicted_classes = []
    predicted_classes.append(classnames[predictions])
    print(predicted_classes)
    #for prediction in predictions:
    #    predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)

if __name__ == '__main__':
    model = load_model()
    data = {
  "body": {
    "userId": "us-east-1:00f1b4f5-365e-42dc-a961-e859f95bd46a",
    "service": "skincancer",
    "invocationId": "XR1599566137961",
    "keys": [['ISIC_0188432.jpg'],
             [0,0.5,0,0,0,0,0,0,0,0,1,0]]
        }
    }

    predictions = run(json.dumps(data))
    predicted_classes = json.loads(predictions)
    print(predicted_classes[0])