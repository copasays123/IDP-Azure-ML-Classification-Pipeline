#import joblib
import os
import fasttext
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Run
import json
import numpy as np
#import os


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'docmodel_v_1.bin')
    model = fasttext.load_model(model_path)

def run(raw_data):
    print('raw---data>', raw_data)
    # try:
    #     data = json.loads(raw_data)['data']
    # except:
    #     data = json.loads(raw_data['data'])
    # print("data--->",data,'---->')
    try:
        labels, probabilities = model.predict(raw_data)
        predicted_label = labels[0]
        confidence = probabilities[0]
        return predicted_label, confidence
    except Exception as e:
        print("Exception ------->", e)
        return e