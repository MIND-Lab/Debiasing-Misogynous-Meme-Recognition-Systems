import pandas as pd
import glob

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