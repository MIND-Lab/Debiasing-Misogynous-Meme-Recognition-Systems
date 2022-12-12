""" 10fold Predictions on training data
Use 10fold models trained on training data (BERT_1_TrainModels_10Fold) to make prediction on associated test fold
This file loads models saved in BERT/models_10Fold folder and make prediction on the associated test fold.

PREDICTIONS ARE NOT EXECUTABLE ON WINDOWS: Issue https://github.com/facebookresearch/mmf/issues/873
    For Windows execution, edit line 77 in file "mmf/mmf/common/test_reporter.py" by substituting ':' with '_'
    line 77 should become: time_format = "%Y-%m-%dT%H_%M_%S"
"""
import os
from Utils import load_data
import shutil
import torch
import gc

saving_folder = './BERT/Predictions/predictions_10Fold'
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

for iteration in range(1, 11):
    # Prediction on trained model
    load_data.upload_yaml(iteration)

    gc.collect()
    torch.cuda.empty_cache()

    model_name = "./BERT/models_10Fold/bert_10Fold_{iteration}.pth".format(iteration=iteration)
    command = "mmf_predict config=projects/hateful_memes/configs/unimodal/bert.yaml\
        model=unimodal_text \
        dataset=hateful_memes \
        run_type=test \
        checkpoint.resume_file={model} \
        checkpoint.resume_pretrained=False".format(model=model_name)
    os.system(command)

    load_data.rename_and_move_predictions(iteration, 'Train', saving_folder)

    # To delete the folder and its content
    shutil.rmtree('./save')
