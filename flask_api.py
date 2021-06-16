import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

####### Libraries related to model ############
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
###############################################

application = Flask(__name__)
model = model = load_model()
diab_dict = { 0:'Negitive', 1:'Positive'}

@application.route('/')
def home():
    return render_template('index.html')

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

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    #print(int_features)
    final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
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

    output = predicted_classes[0]

    return render_template('index.html', prediction_text='The person is predicted as {} for diabetes'.format(output))

@application.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    application.run(debug=True)