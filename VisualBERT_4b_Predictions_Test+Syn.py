""" 10fold Predictions on test and synthetic data
Use models trained on training data (9fold train - 1fold val) to make prediction on associated test and synthetic data.
This file loads models saved in VisualBERT/models folder and make prediction on the datasets.
Predictions made on test are saved in './VisualBERT/Predictions/predictions_test',
predictions made on synthetic are saved in './VisualBERT/Predictions/predictions_syn'

PREDICTIONS ARE NOT EXECUTABLE ON WINDOWS: Issue https://github.com/facebookresearch/mmf/issues/873
    For Windows execution, edit line 77 in file "mmf/mmf/common/test_reporter.py" by substituting ':' with '_'
    line 77 should become: time_format = "%Y-%m-%dT%H_%M_%S"
"""
import os
from Utils import load_data
import shutil
import torch
import gc

saving_folder_test = './VisualBERT/Predictions/predictions_test'
saving_folder_syn = './VisualBERT/Predictions/predictions_syn'
if not os.path.exists(saving_folder_test):
    os.makedirs(saving_folder_test)

if not os.path.exists(saving_folder_syn):
    os.makedirs(saving_folder_syn)

# makes prediction using the 10 modes trained by 3b_TrainModels on different splits of training data
for iteration in range(1, 11):
    load_data.upload_yaml(iteration, 'Test')

    model_name = ".VisualBert/models/visual_bert_model_{iteration}.pth".format(iteration=iteration)
    command = "mmf_predict config=projects/hateful_memes/configs/visual_bert/from_coco.yaml\
        model=visual_bert \
        dataset=hateful_memes \
        run_type=test \
        checkpoint.resume_file={model} \
        checkpoint.resume_pretrained=False".format(model=model_name)
    os.system(command)

    load_data.rename_and_move_predictions(iteration, 'test', saving_folder_test)
    shutil.rmtree('./save')

    load_data.upload_yaml(iteration, 'SYN')
    os.system(command)

    load_data.rename_and_move_predictions(iteration, 'syn', saving_folder_syn)
    shutil.rmtree('./save')

    gc.collect()
    torch.cuda.empty_cache()
