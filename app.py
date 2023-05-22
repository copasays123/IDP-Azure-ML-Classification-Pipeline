from flask import Flask,request, jsonify
import nltk
from azure_ml_pipeline import *
from preprocess_data import *
from config import *
from azureml.core import Dataset, Datastore
from azureml.core import Webservice

app = Flask(__name__)


@app.route('/')
def hello():
    
    return '<h1>Hello, World!</h1>'

@app.route('/uploaddata',methods=['GET'])
def uploaddata():
    #file_path = request.get_json()['path']
    ws = Workspace.from_config()
    from azureml.core import Webservice
    service = Webservice(ws, 'docmodel')
    print(service.get_logs())
    #x = ws.webservices['docmodel'].get_logs()
    print('x====',ws.webservices,'====')
    # default_ds = ws.get_default_datastore()
    # default_ds.upload_files(files=["D:\OneDrive - Coforge Limited\Documents\django_response_extraction.txt"], # Upload the diabetes csv files in /data
    #                    target_path='Experiment_fldr/', # Put it in a folder path in the datastore
    #                    overwrite=True, # Replace existing files of the same name
    #                    show_progress=True)
    return 'done'

@app.route('/train', methods=['POST'])
def training():
    deploy = request.get_json()['deploy']
    ws=create_workspace("IDP-ML-2","IDP-AI","7309ca6b-a0df-4a6e-92ce-cbb04ac1172f")
    print('step-1---------done',ws)
    create_compute_target(ws,"demo-cluster","STANDARD_D2_V2",1,4,120,40)
    print('step-2----------done')
    data_preparation("D:\OneDrive - Coforge Limited\Desktop\model_training_data\IDP_Oversampled_Document_Data_6.csv")
    print('step-3-----------done')
    create_env_to_execute_code_or_training(ws,"demo-cluster",'demo_expirement')
    if deploy:
        model = Model(ws,"docmodel_v_1")
        url = deploy_model(model,ws)
        return url
    else:
        pass
    return 'training done'

@app.route('/pred', methods=['POST'])
def deploy():
    ws = Workspace.from_config()
    service = Webservice(workspace=ws, name="docmodel")
    prediction = service.run(input_data=request.json['data'])
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=5000)
