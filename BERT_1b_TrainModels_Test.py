""" Training and save models on training data
Train 10 models on whole training data (9 fold as training, 1 fold as validation)
then those model will be used to make prediction both on test an on synthetic data to compute bias value.

Trained models are tested in the test fold, and saved in the folder 'models'.

Execution parameters at: https://github.com/facebookresearch/mmf/blob/main/mmf/configs/defaults.yaml
"""

import os
import shutil
import torch
import gc
from Utils import load_data

gc.collect()
torch.cuda.empty_cache()


if not os.path.exists('BERT/models'):
    os.makedirs('BERT/models')

for iteration in range(6, 11):
    # Train del modello
    load_data.upload_yaml(iteration, 'Test')

    torch.cuda.empty_cache()

    os.system("mmf_run config=projects/hateful_memes/configs/unimodal/bert.yaml model=unimodal_text\
        dataset=hateful_memes \
        run_type=train_val \
        training.num_workers=0 \
        evaluation.predict=true \
        training.batch_size=16 \
        training.checkpoint_interval=500\
        training.max_updates=5000 \
        ")


    # rename and move trained model
    model_name = "./BERT/models/bert_model_" + str(iteration) + ".pth"
    shutil.move('./save/unimodal_text_final.pth', model_name)
    shutil.rmtree('./save')

    # clean memory
    gc.collect()
    torch.cuda.empty_cache()

