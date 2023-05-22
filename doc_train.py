from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc, f1_score
from sklearn.model_selection import train_test_split
from azureml.core import Workspace,Environment,Experiment,ScriptRunConfig
import pandas as pd
import fasttext
import joblib
import argparse
from azureml.core import Run
from azureml.core import Dataset, Datastore
from azureml.core import Model
import os

ws = Workspace.from_config()
# datastore = Datastore.get(ws, 'workspaceworkingdirectory')
run = Run.get_context()

def get_prediction(x):
    try:
        return model.predict(x)[0][0]
    except:
        return 'Data Type is not str'

def score(x,y):
    if x == y:
        return 1
    else:
        return 0

try:
    model = fasttext.train_supervised(input='fasttext_train.txt',autotuneValidationFile='fasttext_test.txt')
    validation = model.test('fasttext_test.txt')

    df1 = pd.read_csv(r"validation_sample_1.csv")
    df1['pred_labels'] = df1['text'].apply(get_prediction)
    df1['bool'] = df1.apply(lambda x: score(x.labels, x.pred_labels), axis=1)

    y_actual = df1['labels'].to_list()
    y_predicted = df1['pred_labels'].to_list()

    #classification_report(y_actual,y_predicted)
    confusion_matrix(y_actual,y_predicted)

    acc=accuracy_score(y_actual,y_predicted)
    f1_score = f1_score(y_actual, y_predicted, average='macro')

    #run.log("Accuracy",acc)
    run.log("F1_score", f1_score)
    try:
        if 'outputs' not in os.listdir(os.getcwd()):
            os.makedirs(os.path.join(os.getcwd(),"outputs"))
        else:
            print('pass')
            pass
        model.save_model(os.path.join(os.getcwd(),"outputs","docmodel_v_1.bin"))
        #datastore.upload(src_dir=os.getcwd(),target_path='Experiment_fldr/', overwrite=True)
        path = os.path.join(os.getcwd(),"outputs","docmodel_v_1.bin")
        print('model_deployed----->path', os.listdir(os.path.join(os.getcwd(),"outputs")))
    except Exception as e:
        print('not deployed------>reason', e)
    Model.register(ws, model_path=path, model_name='docmodel_v_1')
    #run.register_model(model_name='docmodel_v1', model_path="outputs/docmodel_v1.bin")
except Exception as e:
    print('some error has been occured --->', e)

#joblib.dump(model, "doc_model.bin")