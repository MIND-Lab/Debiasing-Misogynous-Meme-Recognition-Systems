"""  Features extraction
Images in image_dir that have PNG/JPG/JPEG extensions will have their features extracted
with the following invocation. The output is a collection of .npy and _info.npy files,
containing the features of each image.

Information about requirements and their installation can be found at
    "https://mmf.sh/docs/tutorials/image_feature_extraction"

    The produced features are saved in the "features" folder.
    To be executed it requires:
    - ./Data/TRAINING folder with training images
    - ./Data/TEST folder with test images
    - ./Data/SYNTHETIC folder with synthetic images
"""

import os

folder = './features/training'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = './features/test'
if not os.path.exists(folder):
    os.makedirs(folder)

folder = './features/synthetic'
if not os.path.exists(folder):
    os.makedirs(folder)

os.system("python mmf/tools/scripts/features/extract_features_vmb.py --model_name=X-152 \
        --image_dir=./Data/TRAINING  \
        --output_folder=features/training")

os.system("python mmf/tools/scripts/features/extract_features_vmb.py --model_name=X-152 \
        --image_dir=./Data/TEST  \
        --output_folder=features/test")

os.system("python mmf/tools/scripts/features/extract_features_vmb.py --model_name=X-152 \
        --image_dir=./Data/SYNTHETIC  \
        --output_folder=features/synthetic")
