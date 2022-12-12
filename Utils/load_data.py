import pandas as pd
import glob
import yaml
import os
import time
import pandas as pd

data_path_training = "./Data/training.xls"
data_path_test = "./Data/test.xls"
data_path_syn = "./Data/synthetic.csv"
data_path_syn_identity = "./Data/Syn_identity.csv"

clarifai_path_training = './Data/Clarifai/clarifai_train.csv'
clarifai_path_test = './Data/Clarifai/clarifai_test.csv'
clarifai_path_syn = './Data/Clarifai/clarifai_syn.csv'

azure_path_training = './Data/Azure/azure_training.tsv'
azure_path_test = './Data/Azure/azure_test.tsv'
azure_path_syn = './Data/Azure/azure_syn.tsv'

#__________________________________________Load Data__________________________________________________________

def load_training_data():
    meme_df = pd.read_excel(data_path_training)
    return meme_df

def load_test_data():
    meme_df = pd.read_excel(data_path_test)
    return meme_df

def load_syn_data():
    meme_df = pd.read_csv(data_path_syn, sep='\t')
    return meme_df

def load_syn_identity_data():
    meme_df = pd.read_csv(data_path_syn_identity, sep='\t')
    return meme_df

#__________________________________________CLARIFAI__________________________________________________________
def load_clarifai_training_data():
    """return a dataframe with filename, label, text transcription and clarifai labels (string)"""
    clarifai_df = pd.read_csv(clarifai_path_training)
    for index, row in clarifai_df.iterrows():
      clarifai_df.loc[index, 'clarifai']=clarifai_df.loc[index, 'clarifai'].replace("'", '').replace(",", '').replace("[", '').replace("]", '')
    meme_df = load_training_data()
    meme_df['clarifai'] = ''
    for index, row in meme_df.iterrows():
        meme_df.loc[index, 'clarifai'] = clarifai_df.loc[clarifai_df['id'] == row[0], 'clarifai'].values[0]
    return meme_df

def load_clarifai_test_data():
    """return a dataframe with filename, label, text transcription and clarifai labels (string)"""
    clarifai_df = pd.read_csv(clarifai_path_test)
    for index, row in clarifai_df.iterrows():
      clarifai_df.loc[index, 'clarifai']=clarifai_df.loc[index, 'clarifai'].replace("'", '').replace(",", '').replace("[", '').replace("]", '')
    meme_df = load_test_data()
    meme_df['clarifai'] = ''
    for index, row in meme_df.iterrows():
        meme_df.loc[index, 'clarifai'] = clarifai_df.loc[clarifai_df['id'] == row[0], 'clarifai'].values[0]
    return meme_df

def load_clarifai_syn_data():
    """return a dataframe with filename, label, text transcription and clarifai labels (string)"""
    clarifai_df = pd.read_csv(clarifai_path_syn)
    for index, row in clarifai_df.iterrows():
      clarifai_df.loc[index, 'clarifai']=clarifai_df.loc[index, 'clarifai'].replace("'", '').replace(",", '').replace("[", '').replace("]", '')
    meme_df = load_syn_data()
    meme_df['clarifai'] = ''
    for index, row in meme_df.iterrows():
        meme_df.loc[index, 'clarifai'] = clarifai_df.loc[clarifai_df['id'] == row[0], 'clarifai'].values[0]
    return meme_df

#__________________________________________AZURE__________________________________________________________

def load_azure_caption_training():
    """return a dataframe with filename, label, text transcription and Azure captions (string)"""
    caption_df = pd.read_csv(azure_path_training, sep='\t')
    meme_df = load_training_data()
    meme_df['caption'] = ''
    for index, row in meme_df.iterrows():
        name = row[0].split('.')[0]
        meme_df.loc[index, 'caption'] = caption_df.loc[caption_df['path'] == int(name), 'caption'].values[0]
    return meme_df

def load_azure_caption_test():
    """return a dataframe with filename, label, text transcription and Azure captions (string)"""
    caption_df = pd.read_csv(azure_path_test, sep='\t')
    meme_df = load_test_data()
    meme_df['caption'] = ''
    for index, row in meme_df.iterrows():
        name = row[0].split('.')[0]
        meme_df.loc[index, 'caption'] = caption_df.loc[caption_df['path'] == int(name), 'caption'].values[0]
    return meme_df

def load_azure_caption_syn():
    """return a dataframe with filename, label, text transcription and Azure captions (string)"""
    caption_df = pd.read_csv(azure_path_syn, sep='\t')
    meme_df = load_syn_data()
    meme_df['caption'] = ''
    for index, row in meme_df.iterrows():
        name = row[0].split('.')[0]
        meme_df.loc[index, 'caption'] = caption_df.loc[caption_df['path'] == name, 'caption'].values[0]
    return meme_df


#__________________________________________IDENTITY ELEMENTS__________________________________________________________
identity_terms_path = './Data/IdentityTerms.txt'
identity_tag_path = './Data/Categories.txt'

def read_identity_terms():
    with open(identity_terms_path, 'r') as fd:
        for line in fd:
          Identity_Term_mis = line.split('], ')[0].replace("'", "").strip('][').split(', ')
          Identity_Term_not_mis = line.split('], ')[1].replace("'", "").strip('][').split(', ')
          Identity_Terms = Identity_Term_mis+Identity_Term_not_mis
    return (Identity_Terms)

def read_identity_tags():
    with open(identity_tag_path, 'r') as fd:
        for line in fd:
            tmp = line.lower().split('], ')[0].replace("'", "").strip('][').split(', ')
    #rename adding 'Tag_'
    Identity_Tags = []
    for x in tmp:
        Identity_Tags.append('Tag_' + x)
    return Identity_Tags

def clear_identity_list(identity_list, df):
    """ Take a list of identity elements (tags or temrs), and a dataframe in which every element in the list have a
    corresponding column indicating its presence in the meme.
    Returns a list, subset of identity_list, with the only element to which at least a misogynous and at least a non
     misogynous meme are associated"""
    #At least one misogynous and one not misogynous per tag:
    to_remove=[]
    for tag in identity_list:
        if len(df.loc[df[tag]==1,'misogynous'].value_counts())<2:
            to_remove.append(tag)
    for tag in to_remove:
        identity_list.remove(tag)
    return identity_list

def read_clear_identity_terms():
    identity_terms = read_identity_terms()
    df = load_syn_identity_data()
    identity_terms = clear_identity_list(identity_terms, df)
    return identity_terms

def read_clear_identity_tags():
    identity_tags = read_identity_tags()
    df = load_syn_identity_data()
    identity_tags = clear_identity_list(identity_tags, df)
    return identity_tags

# __________________________________________VISUALBERT__________________________________________________________


def upload_yaml(iteration=1, test_data='Train'):
    """ This function edit the 'hateful_memes/defaults.yaml' file in order to specify path to the data needed
    for the training and testing phase.
    The test_data variable is used to define the dataset to test on. There are three types of training-test:
        1. a 10-fold execution on training data (8Fold train, 1fold val, 1fold test)
        2. a 10-fold execution on training data (9fold train, 1fold val) tested on Test data
        3. a 10-fold execution on training data (9fold train, 1fold val) tested on Synthetic data

    :param iteration: number of iteration (10-fold on training data)
    :param test_data: to chose between the following:
        - 'Train' (default value): for 10Fold on training data
        - 'SYN': train on training data (9fold train, 1fold val), test on synthetic data
        - 'Test' or other: train on training data (9fold train, 1 val), test on test data
    """

    train_images = os.path.abspath('./Data/TRAINING')
    train_features = os.path.abspath('./features/training')
    train_10Fold_annotations = os.path.abspath('./Annotations/10Fold_train')
    train_annotations = os.path.abspath('./Annotations/train')

    test_images = os.path.abspath('./Data/TEST')
    test_features = os.path.abspath('./features/test')
    test_annotations = os.path.abspath('./Annotations/test_complete.jsonl')

    sy_images = os.path.abspath('./Data/SYNTHETIC')
    sy_features = os.path.abspath('./features/synthetic')
    sy_annotations = os.path.abspath('./Annotations/synthetic')

    if test_data == 'Train':
        train_annotations = train_10Fold_annotations
        test_images = train_images
        test_features = train_features
        test_annotations = train_annotations
    elif test_data == 'SYN' or test_data == 'SYN_BO':
        test_images = sy_images
        test_features = sy_features
        test_annotations = sy_annotations

    with open('./mmf/mmf/configs/datasets/hateful_memes/defaults.yaml') as f:
        list_doc = yaml.safe_load(f)

    # images
    list_doc['dataset_config']['hateful_memes']['images']['train'] = train_images
    list_doc['dataset_config']['hateful_memes']['images']['val'] = train_images
    list_doc['dataset_config']['hateful_memes']['images']['test'] = test_images

    # features
    list_doc['dataset_config']['hateful_memes']['features']['train'] = train_features
    list_doc['dataset_config']['hateful_memes']['features']['val'] = train_features
    list_doc['dataset_config']['hateful_memes']['features']['test'] = test_features

    # annotations
    list_doc['dataset_config']['hateful_memes']['annotations']['train'] = train_annotations + r'\train_' + str(
        iteration) + '.jsonl'
    list_doc['dataset_config']['hateful_memes']['annotations']['val'] = train_annotations + r'\val_' + str(
        iteration) + '.jsonl'

    if test_data == 'Train' or test_data == 'SYN':
        list_doc['dataset_config']['hateful_memes']['annotations']['test'] = test_annotations + r'\test_' + str(
            iteration) + '.jsonl'
    elif test_data == 'SYN_BO':
        list_doc['dataset_config']['hateful_memes']['annotations']['test'] = test_annotations + r'\train_' + str(
            iteration) + '.jsonl'
    else:
        list_doc['dataset_config']['hateful_memes']['annotations']['test'] = test_annotations

    with open('./mmf/mmf/configs/datasets/hateful_memes/defaults.yaml', "w") as f:
        yaml.dump(list_doc, f)


def rename_and_move_predictions(iteration, data_type, output_path, acronym='pred'):
    """rename predictions file made by visual bert.
    The file, renamed as ex.'BO_Test_1.csv' will be also moved from the 'save' folder
    to the 'output_path' folder.
    Time is used to disambiguate older predictions

    :param iteration: iteration number (int)
    :param data_type: string to define if prediction refers to train (10Fold), Test or synthetic data
    :param output_path: path to saving folder
    :param acronym: prefix to use in filename (default = 'pred')

    predictions made by VisualBert are moved from 'save' folder to the output_path folder, and rename according to
    information (data_type, output_path and acronym) in input.

    :return new file path
    """

    name = ''
    # based on the assumption that only one .csv file is present in 'save' folder
    for root, dirs, files in os.walk('./save'):
        for file in files:
            if file.endswith(".csv"):
                name = acronym + '_' + data_type + '_' + str(iteration) + '_' + str(int(time.time())) + '.csv'
                name = os.path.join(output_path, name)
                os.rename(os.path.join(root, file), name)
    return name

#__________________________________________ Other __________________________________________________________

def shuffle_syn(syn_unshuffled, column_names):
    """shuffle synthetic dataset in input according to the order in file '.\Data\synthetic_shuffled.csv'
    
    :param syn_unshuffled: dataframe to reorder
    :param column_names: name of the column containing synthetic memes names
    :rettrn dataframe shuffled
    """
    shuff_df= pd.read_csv('.\Data\synthetic_shuffled.csv', sep='\t')
    shuff_df.set_index('file_name', inplace=True)
    syn_unshuffled.set_index(column_names, inplace=True)
    syn_unshuffled = syn_unshuffled.reindex(shuff_df.index)
    syn_unshuffled.reset_index(inplace=True)
    return syn_unshuffled

def load_predictions_data(path_predictions):
    """
    This function loads predictions obtained after Bayesian Optimization process.
    In particular, reads predictions made on test and on synthetic data and merge those data with memes' information
    and real label.

    :param path_predictions: path to the folder that contains predictions on syn and on test data
    :return: two dataframe, respectively with test and sinthetic data and predictions.
    """
    test_df = load_test_data()
    for fname in glob.glob(path_predictions + 'BO_Final_Test_*.csv'):
        pred = pd.read_csv(fname, sep='\t').drop(columns=['misogynous'])
        test_df = pd.merge(test_df, pred, on="file_name")

    syn_df = load_syn_identity_data()
    for fname in glob.glob(path_predictions + 'BO_Final_SYN_*.csv'):
        pred = pd.read_csv(fname, sep='\t').drop(columns='misogynous')
        syn_df = pd.merge(syn_df, pred, on="file_name")

    return test_df, syn_df