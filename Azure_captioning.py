"""
Azure Captioning
Extract image captions with Azure API.
(installed via 'pip install --upgrade azure-cognitiveservices-vision-computervision').
Image captions are obtained through Azure Cognitive services.

To execute respectively on training, test or synthetic data, please uncomment the corresponding data path and insert
subscription key and endpoint at line 19-20.
"""

import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import time
import json
import pandas as pd

subscription_key = "INSERT KEY"
endpoint = "INSERT ENDPOINT"

# Create the computer vision client
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

# _______________________________________Tag Training Data ___________________________________________
image_path = './Data/TRAINING'
json_path = './Data/Azure/azure_training_captions.json'
tsv_path = './Data/Azure/azure_training.tsv'
# _______________________________________Tag Test Data ___________________________________________
"""
image_path = './Data/TEST'
json_path = './Data/Azure/azure_test_captions.json'
tsv_path = './Data/Azure/azure_test.tsv
"""
# _______________________________________Tag Synthetic Data ___________________________________________
"""
image_path = './Data/SYNTHETIC'
json_path = './Data/Azure/azure_syn_captions.json'
tsv_path = './Data/Azure/azure_syn.tsv'
"""

if not os.path.exists('./Data/Azure/'):
    os.makedirs('./Data/Azure/')

dump_json = []
root_dir = image_path
images = os.listdir(root_dir)

for image_name in images:
    time.sleep(3)
    image_id = int(image_name.split('.')[0])

    with open(root_dir + image_name, "rb") as image:
        description_results = computervision_client.describe_image_in_stream(image)

    # Get the first caption (description) from the response
    if len(description_results.captions) == 0:
        image_caption = "No description detected."
    else:
        image_caption = description_results.captions[0].text
        dump_json.append({'path': image_id, "caption": image_caption, "objects": description_results.tags})

with open(json_path, 'w') as f:
    json.dump(dump_json, f)

f = open(json_path)
caption_dict = json.load(f)
df = pd.DataFrame.from_dict(caption_dict)
df.to_csv(tsv_path, index=False, header=True, sep='\t')
