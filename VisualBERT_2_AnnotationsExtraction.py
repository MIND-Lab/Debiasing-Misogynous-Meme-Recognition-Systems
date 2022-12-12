""" Annotations extractions
Create annotations in order to train model in 10-Fold on training data (9fold training -1fold validation)
  and test both on Test and on synthetic data.

  The produced Annotations are saved in the folder './Annotations' in jsonl format.
  To be executed it requires:
    - Data/training.xls
    - Data/test.xls
    - Data/synthetic.csv
  Those files should have the following columns names: 'file_name', 'misogynous', 'Text Transcription'
  """

import pandas as pd
import json
from sklearn.model_selection import KFold
import os
from Utils import load_data


# ________________________________________Utils ___________________________________________________________________


def train_to_json(train, val, test, iteration, folder_name):
    """
    :param train: dataframe with training data
    :param val: dataframe with validation data
    :param test: dataframe with test data
    :param iteration: iteration number (for saving purpose)

    This function create json files for <train, val, test> for the single 10-fold split.
    Those files are saved in the folder "Annotations/train/
    """
    temp = []

    for index, _ in train.iterrows():
        thisdict = {
            "id": int(train.loc[index, 'unique_number']),
            "img": str(train.loc[index, 'file_name']) + ".jpg",
            "label": int(train.loc[index, 'misogynous']),
            "text": train.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)
    x = json.dumps(temp)
    name = folder + folder_name + "/train_" + str(iteration) + ".json"
    with open(name, 'w') as fp:
        fp.write(x)

    temp = []

    for index, _ in val.iterrows():
        thisdict = {
            "id": int(val.loc[index, 'unique_number']),
            "img": str(val.loc[index, 'file_name']) + ".jpg",
            "label": int(val.loc[index, 'misogynous']),
            "text": val.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)
    x = json.dumps(temp)
    name = folder + folder_name + "/val_" + str(iteration) + ".json"
    with open(name, 'w') as fp:
        fp.write(x)

    temp = []

    for index, row in test.iterrows():
        thisdict = {
            "id": int(test.loc[index, 'unique_number']),
            "img": str(test.loc[index, 'file_name']) + ".jpg",
            "text": test.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)

    x = json.dumps(temp)
    name = folder + folder_name + "/test_" + str(iteration) + ".json"
    with open(name, 'w') as fp:
        fp.write(x)

def syn_10fold_to_json(train, val, iteration, folder_name):
    """
    :param train: dataframe with training data
    :param val: dataframe with validation data
    :param test: dataframe with test data
    :param iteration: iteration number (for saving purpose)

    Variation of the previous method for synthetic data split
    """
    temp = []

    for index, _ in train.iterrows():
        thisdict = {
            "id": int(train.loc[index, 'unique_number']),
            "img": str(train.loc[index, 'file_name']) + ".jpg",
            #"label": int(train.loc[index, 'misogynous']),
            "text": train.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)
    x = json.dumps(temp)
    name = folder + folder_name + "/train_" + str(iteration) + ".json"
    with open(name, 'w') as fp:
        fp.write(x)

    temp = []

    for index, _ in val.iterrows():
        thisdict = {
            "id": int(val.loc[index, 'unique_number']),
            "img": str(val.loc[index, 'file_name']) + ".jpg",
            "label": int(val.loc[index, 'misogynous']),
            "text": val.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)
    x = json.dumps(temp)
    name = folder + folder_name + "/test_" + str(iteration) + ".json"
    with open(name, 'w') as fp:
        fp.write(x)

    temp = []

def test_to_json(test):
    """
    :param test: dataset with test data

    This function create a json file to test on those data
    """
    temp = []

    for index, _ in test.iterrows():
        thisdict = {
            "id": int(test.loc[index, 'unique_number']),
            "img": str(test.loc[index, 'file_name']) + ".jpg",
            "text": test.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)

    x = json.dumps(temp)
    name = folder + "/test_complete.json"
    with open(name, 'w') as fp:
        fp.write(x)


def syn_to_json(test):
    """
    :param test: dataframe with synthetic data

    This function create a json file to make predictions on those data
    """
    temp = []

    for index, _ in test.iterrows():
        thisdict = {
            "id": int(test.loc[index, 'unique_number']),
            "img": str(test.loc[index, 'file_name']) + ".jpg",
            "text": test.loc[index, 'Text Transcription']
        }
        temp.append(thisdict)

    x = json.dumps(temp)
    name = folder + "/syn_complete.json"
    with open(name, 'w') as fp:
        fp.write(x)


# _________________________________________Main_________________________________________

folder = './Annotations'
if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + '/train'):
    os.makedirs(folder + '/train')

if not os.path.exists(folder + '/10Fold_train'):
    os.makedirs(folder + '/10Fold_train')
    
if not os.path.exists(folder + '/synthetic'):
    os.makedirs(folder + '/synthetic')

meme_df = load_data.load_training_data()

for index, row in meme_df.iterrows():
    meme_df.loc[index, 'unique_number'] = int(meme_df.loc[index, 'file_name'].split('.')[0])

# Annotations files <train-val-test> (.json) are created in 10-fold to allow 10-fold execution on training dataset
kf = KFold(n_splits=10, shuffle=False)

iteration = 0

for train, val in kf.split(meme_df):  # split into train and test
    test = []
    train_to_json(meme_df.iloc[train, :], meme_df.iloc[val, :], meme_df.iloc[test, :], iteration + 1, '/train')

    # last training fold is used as validation
    test = val
    if iteration == 0:
        i = 8
    else:
        i = iteration - 1
    a, b = list(KFold(n_splits=9).split(train))[i]
    val = train[b]
    train = train[a]
    train_to_json(meme_df.iloc[train, :], meme_df.iloc[val, :], meme_df.iloc[test, :], iteration + 1, '/10Fold_train')
    iteration = iteration + 1

"""Annotations to test on Test Set"""

meme_df = load_data.load_test_data()

for index, row in meme_df.iterrows():
    meme_df.loc[index, 'unique_number'] = int(meme_df.loc[index, 'file_name'].split('.')[0])

test_to_json(meme_df)

json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]



""" drop empty files (train folder contains 9fold train, 1 fold val, test files are empty)"""
for file_name in os.listdir(folder + "/train"):
    if file_name.endswith(".json") and file_name.startswith("test"):
        print(os.path.join(folder + "/train/", file_name))
        os.remove(os.path.join(folder + "/train/", file_name))



"""Annotations to test on Synthetic data"""
csv_path = "./Data/synthetic.csv"
syn_df = pd.read_csv(csv_path, usecols=['file_name', 'misogynous', 'Text Transcription'], sep='\t')

for index, row in syn_df.iterrows():
    syn_df.loc[index, 'unique_number'] = int(syn_df.loc[index, 'file_name'].split('.')[0].split('_')[1])

# Annotations files <train-val-test> (.json) are created in 10-fold to allow 10-fold execution on training dataset
kf = KFold(n_splits=10, shuffle=False)

iteration = 0

for train, val in kf.split(syn_df):  # split into train and test
    syn_10fold_to_json(syn_df.iloc[train, :], syn_df.iloc[val, :], iteration + 1, '/synthetic')

    # last training fold is used as validation
    test = val
    if iteration == 0:
        i = 8
    else:
        i = iteration - 1
    a, b = list(KFold(n_splits=9).split(train))[i]
    val = train[b]
    train = train[a]
    #train_to_json(syn_df.iloc[train, :], syn_df.iloc[val, :], syn_df.iloc[test, :], iteration + 1, '/10Fold_synthetic')
    iteration = iteration + 1


""" drop empty files (train folder contains 9fold train, 1 fold val, test files are empty)"""
for file_name in os.listdir(folder + "/train"):
    if file_name.endswith(".json") and file_name.startswith("test"):
        print(os.path.join(folder + "/train/", file_name))
        os.remove(os.path.join(folder + "/train/", file_name))
            
""" json to jsonl"""
for root, dirs, files in os.walk(folder):
    for file_name in files:
        if file_name.endswith((".json")):
            with open(os.path.join(root, file_name), "r") as read_file:
                JSON_file = json.load(read_file)

                name = file_name.split('.')[0] + '.jsonl'

                with open(os.path.join(root, name), 'w') as outfile:
                    for entry in JSON_file:
                        json.dump(entry, outfile)
                        outfile.write('\n')
            os.remove(os.path.join(root, file_name))
