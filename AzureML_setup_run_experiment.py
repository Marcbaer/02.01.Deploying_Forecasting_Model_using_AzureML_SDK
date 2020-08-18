# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:28:36 2020

@author: Marc
"""

'''01. Connect to ML Workspace'''
from azureml.core import Workspace, Datastore, Dataset

ws = Workspace.from_config()

'''02. List all compute targets'''
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)

#chose compute target
compute_name='comptarget'

'''03. Register Lotka Volterra Data'''
blob_ds = Datastore.get(ws, datastore_name='blob_data_marc')
file_ds = Dataset.File.from_files(path=(blob_ds, 'Lotka_Volterra2.p'))
file_ds = file_ds.register(workspace=ws, name='Lotka_Volterra_Data')

'''04. Create New Training Environment'''
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('lotkavolterra_environment_marc')
deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy','tensorflow','matplotlib'],
                                pip_packages=['azureml-defaults'])
env.python.conda_dependencies = deps

#Use existing environment
# env = Environment.from_existing_conda_environment(name='lotkavolterra_environment_marc',
#                                                   conda_environment_name='py_env')

env.register(workspace=ws)

#View registered envs
env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:',env_name)
    
    
'''04. Create and Run New Experiment'''
from azureml.core import Experiment
from azureml.train.estimator import Estimator

# Create a folder for the experiment files
experiment_folder = './'

estimator = Estimator(source_directory=experiment_folder,
                      entry_script='Train_LSTM_Experiment.py',
                      environment_definition=env,
                      compute_target=compute_name)

# submit the experiment
experiment = Experiment(workspace = ws, name = 'LotkaVolterra-experiment')
run = experiment.submit(config=estimator)
#RunDetails(run).show()
run.wait_for_completion()

# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)

'''05. Check Experiment Output'''

LV_experiment = ws.experiments['LotkaVolterra-experiment']
for logged_run in LV_experiment.get_runs():
    print('Run ID:', logged_run.id)
    metrics = logged_run.get_metrics()
    for key in metrics.keys():
        print('-', key, metrics.get(key))
        
        
        
        
        