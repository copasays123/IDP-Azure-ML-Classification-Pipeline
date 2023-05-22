import azure.core
from azureml.core import Workspace,Environment,Experiment,ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from config import *


# Now create Workspace
def create_workspace(ws_name, res_name, subs_id):
    try:
        ws=Workspace.from_config()
        print('Workspace is already exist')
    except:
        ws=Workspace.create(ws_name, 
                        resource_group=res_name,
                        create_resource_group=True,
                        subscription_id=subs_id,
                        location="East US")
        ws.write_config('.azureml')
        print('New Workspace created')
    return ws
    

# Create Compute Target
def create_compute_target(ws, aml_compute_target, vm_size=None, min_nodes=None, max_nodes=None, idle_seconds=None,timeout=None):
    try:
        aml_compute = AmlCompute(ws, aml_compute_target)
        print("This Compute Target already exist.")
    except ComputeTargetException:
        print("creating new compute target :",aml_compute_target)
        
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                    min_nodes = min_nodes, 
                                                                    max_nodes = max_nodes,
                                                idle_seconds_before_scaledown=idle_seconds)    
        aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
        aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=timeout)
        
        print("Azure Machine Learning Compute attached now")
        return 'done'

def create_env_to_execute_code_or_training(ws, aml_compute_target, experiment_name=None):

    # Create Experiment
    exp = Experiment(ws,experiment_name)

    # Create environment to execute your code
    env = Environment.from_conda_specification(name="azure_ml",file_path="./envfile.yml")
    config=ScriptRunConfig(source_directory=".",script="doc_train.py",compute_target=aml_compute_target,environment=env)
    try:
        execution=exp.submit(config)
        execution.wait_for_completion(show_output=True)
    except Exception as e:
        print("Exception ====>",e)
    return 'training done !!!'

#register model
def register_model(ws, model_nm, path):
    model = Model.register(ws, model_path=path, model_name=model_nm)
    return Model(ws,model_nm)

#deploy model
def deploy_model(model,ws):
    myenv=Environment(name="demo-env")
    conda_packages = ['numpy']
    pip_packages = ['azureml-sdk','azureml-defaults','scikit-learn','fasttext-wheel','nltk','gensim','pandas']
    mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages, python_version='3.8.5')
    myenv.python.conda_dependencies=mycondaenv
    myenv.register(workspace=ws)
    inference_config = InferenceConfig(entry_script='score.py',source_directory='.',environment=myenv)

    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,memory_gb=1)

    #from azureml.core.webservice import LocalWebservice

    #localconfig = LocalWebservice.deploy_configuration(port=6789)

    service = Model.deploy(ws,
                           'docmodel',
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aciconfig,
                           overwrite=True)
    #print("logs1--->",service.get_logs(),"1====>")
    service.wait_for_deployment(show_output=True)
    try:
        print("logs--->",service.get_logs())
    except Exception as e:
        print("Exception----------->",e)
    #print("logs--->",service.get_logs(),"====>")
    url = service.scoring_uri
    return url

