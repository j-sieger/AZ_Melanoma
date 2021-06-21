import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

####### Libraries related to model ############
import json
import numpy as np
import torch
from melanoma_service.model_eff import load_model
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
UPLOAD_FOLDER = './upload'

@application.route('/')
def home():
    return render_template('index.html')

# Called when a request is received
def run(data):

    # Get a prediction from the model
    z_val = model(data)
    predictions = torch.sigmoid(z_val)
    predictions =  predictions.type(torch.int).item()
    print(predictions)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['Not-Melanoma', 'Melanoma']
    predicted_classes = []
    predicted_classes.append(classnames[predictions])
    return json.dumps(predicted_classes)

def input_data(r):
    features = []
    features.append(float(r.form['Sex']))
    features.append(float(r.form['Age']))
    features.append(float(r.form['AnteriorTorso']))
    features.append(float(r.form['HeadNeck']))
    features.append(float(r.form['LateralTorso']))
    features.append(float(r.form['LowerExtremity']))
    features.append(float(r.form['Genitials']))
    features.append(float(r.form['Soles']))
    features.append(float(r.form['PosteriorTorso']))
    features.append(float(r.form['Torso']))
    features.append(float(r.form['UpperExtremity']))
    features.append(float(r.form['NAN']))
    return features

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    meta_features = input_data(request)
    img_input    = request.files['file_img']
    path = os.path.join(UPLOAD_FOLDER, img_input.filename)
    img_input.save(path)
    
    x = cv2.imread(path)
    x = test_transform(x).unsqueeze(0)
    meta = torch.tensor(meta_features).unsqueeze(0)
    meta = meta.type(torch.FloatTensor)
    
    final_features = (x,meta)
    predictions = run(final_features)
    predicted_classes = json.loads(predictions)

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