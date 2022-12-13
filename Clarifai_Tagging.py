"""
Clarifai Tagging
Tag images with Clarifai API. (installed via 'pip install clarifai-grpc').
Images are tagged according to 14 selected categories stored in the file 'Data/Categories.txt'.
At the end of execution csv files with tags information are saved in './Data/Clarifai' folder.

To execute respectively on training, test or synthetic data, please uncomment the corresponding data path and insert
Clarifai key at line 144.
NB: Clarifai API allow a maximum of 1000 calls for month.
Produced annotations have been manually joined.
"""

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import pandas as pd
import os
import json
from Utils import load_data

# _______________________________________Tag Training Data ___________________________________________
json_path = "./Data/Clarifai/image_data_train.json"
csv_path = './Data/training.xls'
image_path = './Data/TRAINING'
csv_out_path = 'Data/Clarifai/clarifai_train.csv'
# _______________________________________Tag Test Data ___________________________________________
"""
json_path = "./Data/Clarifai/image_data_test.json"
test_csv_path = './Data/test.xls'
image_path = './Data/TEST'
csv_out_path='Data/Clarifai/clarifai_test.csv'
"""
# _______________________________________Tag Synthetic Data ___________________________________________
"""
json_path = "./Data/Clarifai/image_data_syn.json"
csv_path = './Data/synthetic.csv'
image_path = './Data/SYNTHETIC'
csv_out_path='Data/Clarifai/clarifai_syn.csv'
"""

if not os.path.exists('./Data/Clarifai'):
    os.makedirs('./Data/Clarifai')


# ___________________________________________Utils_____________________________________________________
def get_labels(file_bytes):
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            model_id='aaa03c23b3724a16a56b629203edc62c',
            #version_id="{THE_MODEL_VERSION_ID}",  # This is optional. Defaults to the latest model version.
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ],
            model=resources_pb2.Model(
                output_info=resources_pb2.OutputInfo(
                    output_config=resources_pb2.OutputConfig(
                        select_concepts=[
                            # When selecting concepts, value is ignored, so no need to specify it.
                            resources_pb2.Concept(name="animal"),
                            resources_pb2.Concept(name="broom"),
                            resources_pb2.Concept(name="car"),
                            resources_pb2.Concept(name="illustration"), #cartoon
                            resources_pb2.Concept(name="cat"),
                            resources_pb2.Concept(name="child"),
                            resources_pb2.Concept(name="dishware"), #crockery
                            resources_pb2.Concept(name="glass"), #crockery
                            resources_pb2.Concept(name="flatware"), #crockery
                            resources_pb2.Concept(name="dishwasher"), 
                            resources_pb2.Concept(name="dog"),
                                                      
                            resources_pb2.Concept(name="kitchenware"),#kitchenUtensil
                            resources_pb2.Concept(name="cookware"),#kitchenUtensil
                            
                            resources_pb2.Concept(name="oven"), #kitchen
                            resources_pb2.Concept(name="stove"), #kitchen
                            resources_pb2.Concept(name="refrigerator"), #kitchen
                            resources_pb2.Concept(name="cabinet"), #kitchen
                            
                            resources_pb2.Concept(name="man"),
                            
                            resources_pb2.Concept(name="nude"),#nudity
                            resources_pb2.Concept(name="topless"),#nudity
                            resources_pb2.Concept(name="nudist"),#nudity
                            resources_pb2.Concept(name="bikini"),#nudity
                            
                            resources_pb2.Concept(name="woman"),
                        ]
                    )
                )
            )
            
        ),
        metadata=metadata
    )    
    
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
        
    # Since we have one input, one output will exist here.
    output = post_model_outputs_response.outputs[0]
    return output


def category_mapping(output):
    results = dict((str.lower(el),0) for el in image_categories)

    for concept in output.data.concepts:
        for categoria in image_categories:
            if concept.name == str.lower(categoria):
                results[concept.name]= concept.value
            elif concept.name == 'illustration':
                results['cartoon']=concept.value
            elif concept.name in ['dishware', 'glass', 'flatware'] and concept.value > results['crockery']:
                results['crockery']=concept.value
            elif concept.name in ['oven','stove', 'refrigerator', 'cabinet'] and concept.value > results['kitchen']:
                results['kitchen']=concept.value
            elif concept.name in ['nude','topless', 'nudist', 'bikini'] and concept.value > results['nudity']:
                results['nudity']=concept.value
            elif concept.name in ['kitchenware','cookware'] and concept.value > results['kitchenutensil']:
                results['kitchenutensil']=concept.value
    return results


def save_data(json_path, *runs):
    with open(json_path, "a") as f:
        json.dump(list(runs)[0], f)
        f.write('\n')


# ___________________________________________CLARIFAI_____________________________________________________
# For every image in the dataframe get image categories.
# it creates a temporary dictionary with image data and labels.
# Set a threshold to select image tags, then write dictionary to csv.

channel = ClarifaiChannel.get_json_channel()
stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_json_channel())  # the HTTPS+JSON channel

metadata = (('authorization', 'INSERT Key '),)

get_model_response = stub.GetModelOutputInfo(
    service_pb2.GetModelRequest(model_id="aaa03c23b3724a16a56b629203edc62c"),
    metadata=metadata
)

if get_model_response.status.code != status_code_pb2.SUCCESS:
    raise Exception("Get model failed, status: " + get_model_response.status.description)

model = get_model_response.model

# _______________________________________________LOAD DATA________________________________________
image_categories = load_data.read_identity_tags()

image_df = pd.read_excel(csv_path, usecols=['file_name'])
path = image_path + '/'
image_df['image_path'] = path + image_df['file_name']

# _______________________________________________Image Prediction_________________________________________________
"""
Clarifai key allows 1,000 free operations monthly.
The following line of code allow to restore the annotations process by inserting the number of already annotated meme.
image_df = image_df[1000:]
"""

"""
Save percentages of confidence for every tag in a json file.
Stops after the firsts 1000 images because of Clarifai limitations
"""
for index, row in image_df[:1000].iterrows():
    image_data = {'id': image_df.loc[index, 'file_name'],
                  'url': image_df.loc[index, 'image_path'],
                  'label': []}

    # get image labels
    with open(image_df.loc[index, 'image_path'], "rb") as f:
        file_bytes = f.read()

    try:
        output = get_labels(file_bytes)
        results = category_mapping(output)
    except:
        print('Amount of Clarifai monthly calls reached')
        break

    category_dic = dict((el, 0) for el in results)
    for categoria in results:
        category_dic[categoria] = results[categoria]

    image_data['label'].append(str(category_dic))

    # save to JSON
    save_data(json_path, image_data)

# _______________________________________________Json to csv_________________________________________________

clarifai = pd.read_json(json_path, lines=True)

# save results on CSV
csv = clarifai['id'].to_frame()
csv['clarifai'] = None

rowsC = clarifai.shape[0]
soglia = 0.85

for i in range(rowsC):
    clarifai_ID = clarifai['id'][i]
    clarifai_labels_dic = json.loads(clarifai['label'][i][0].replace("\'", "\""))
    clarifai_labels = []

    for x in clarifai_labels_dic.keys():
        if clarifai_labels_dic[x] > soglia:
            clarifai_labels.append(x)
    soglia2 = soglia
    while (not clarifai_labels):
        for x in clarifai_labels_dic.keys():
            if clarifai_labels_dic[x]>soglia2:
                clarifai_labels.append(x)
        soglia2=soglia2-0.1

    if clarifai_labels is None:
        clarifai_labels = []
    if clarifai_labels:
        clarifai_labels = [x.lower() for x in clarifai_labels]

    csv.loc[csv['id'] == clarifai_ID, 'clarifai'] = str(clarifai_labels)

csv.to_csv(csv_out_path, index=False)
