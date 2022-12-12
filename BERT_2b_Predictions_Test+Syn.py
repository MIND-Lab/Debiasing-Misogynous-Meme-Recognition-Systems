""" 10fold Predictions on test and synthetic data
Use models trained on training data (9fold train - 1fold val) to make prediction on associated test and synthetic data.
This file loads models saved in BERT/models folder and make prediction on the datasets.
Predictions made on test are saved in './BERT/Predictions/predictions_test',
predictions made on synthetic are saved in './BERT/Predictions/predictions_syn'

PREDICTIONS ARE NOT EXECUTABLE ON WINDOWS: Issue https://github.com/facebookresearch/mmf/issues/873
    For Windows execution, edit line 77 in file "mmf/mmf/common/test_reporter.py" by substituting ':' with '_'
    line 77 should become: time_format = "%Y-%m-%dT%H_%M_%S"
"""
import os
from Utils import load_data
import shutil
import torch
import gc


saving_folder_syn = './BERT/Predictions/predictions_syn'

if not os.path.exists(saving_folder_syn):
    os.makedirs(saving_folder_syn)

saving_folder_test = './BERT/Predictions/predictions_test'
if not os.path.exists(saving_folder_test):
    os.makedirs(saving_folder_test)

# makes prediction using the 10 modes trained by 3b_TrainModels on different splits of training data
for iteration in range(1, 11):

    model_name = "./BERT/models/bert_model_{iteration}.pth".format(iteration=iteration)
    command = "mmf_predict config=projects/hateful_memes/configs/unimodal/bert.yaml\
        model=unimodal_text \
        dataset=hateful_memes \
        run_type=test \
        checkpoint.resume_file={model} \
        checkpoint.resume_pretrained=False".format(model=model_name)

    load_data.upload_yaml(iteration, 'SYN')
    os.system(command)

    load_data.rename_and_move_predictions(iteration, 'syn', saving_folder_syn)
    shutil.rmtree('./save')

    load_data.upload_yaml(iteration, 'Test')
    os.system(command)

    load_data.rename_and_move_predictions(iteration, 'test', saving_folder_test)
    shutil.rmtree('./save')

    gc.collect()
    torch.cuda.empty_cache()