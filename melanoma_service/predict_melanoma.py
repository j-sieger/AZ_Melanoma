import json
import joblib
import numpy as np
from azureml.core.model import Model
import torch
#from model_eff import load_model
from melanoma_service.model_eff import load_model
#from predict import predict
from time import time

from azureml.core.model import Model
from efficientnet_pytorch import EfficientNet
import torchtoolbox.transform as transforms
import torch.nn as nn
import torch
import cv2

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    #model_path = Model.get_model_path('Melanoma_Classify')
    #model = joblib.load(model_path)
    
    #arch = EfficientNet.from_pretrained('efficientnet-b1')  # Going to use efficientnet-b1 NN architecture
    #model = Net(arch=arch, n_meta_features=12)
    #model_path = Model.get_model_path('Melanoma_Classify')
    #model = torch.load(model_path, map_location=torch.device('cpu'))
    #model.eval()
    
    model = load_model()
  
def data_format(data):
    
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    print("jani",data[0][0])
    x = cv2.imread(data[0][0])
    x = test_transform(x).unsqueeze(0)
    meta = torch.tensor(data[1]).unsqueeze(0)
    meta = meta.type(torch.FloatTensor)

    return (x,meta)

# Called when a request is received
def run(raw_data):
    data = data_format(json.loads(raw_data)['body']['keys'])
    # Get the input data as a numpy array
    #data = np.array(json.loads(raw_data)['body']['keys'][1])
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
