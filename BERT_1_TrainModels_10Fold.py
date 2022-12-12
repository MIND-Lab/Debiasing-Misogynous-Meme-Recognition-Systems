""" Training and save models in 10Fold on training data
Training data are split in 10Fold and used to train models. Trained models are tested in the test fold, and saved in
the folder 'models_10Fold'.

Execution parameters at: https://github.com/facebookresearch/mmf/blob/main/mmf/configs/defaults.yaml
Eventually, uncomment code to restore checkpoints.
"""

import os
import shutil
import torch
import gc
from Utils import load_data


if not os.path.exists('BERT/models_10Fold'):
    os.makedirs('BERT/models_10Fold')

for iteration in range(1, 11):
    # Train del modello
    load_data.upload_yaml(iteration)
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
    model_name = "./BERT/models_10Fold/bert_10Fold_" + str(iteration) + ".pth"
    shutil.move('./save/unimodal_text_final.pth', model_name)
    shutil.rmtree('./save')

    # clean memory
    gc.collect()
    torch.cuda.empty_cache()

