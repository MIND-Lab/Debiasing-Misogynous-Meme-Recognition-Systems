# IMPORT
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import gc
import random

identity_tag_path = './Data/Categories.txt'


def use_preprocessing(df, column):
    """Compute the embedding via the Universal Sentence Encode algorithm
    for every sentence in the given column
    Args:
        df: Dataframe
        column: column name to identify data to process
    """
    # Universal Sentence Encoder
    tf.compat.v1.disable_eager_execution()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed"
    embed = hub.Module(module_url)

    dfs = np.array_split(df, 10)

    # Split in 10 call because during embedding creation an error occur after 47900 steps.
    text_embeddings = []
    with tf.compat.v1.Session(config=config) as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

        for x in dfs:
            # x_data_matches = pd.Series([])
            text_embedding = session.run(embed(list(x[column])))
            text_embeddings = text_embeddings + np.array(text_embedding).tolist()
        session.close()

        del text_embedding
        del dfs
        gc.collect()

    return text_embeddings


def elaborate_input(data, input_columns, label_column):
    """ return two dataframe (obtained as a subset of the input one): 
    one with the columns that represent the input of the model and
    one with the label column
    Args:
        data: dataframe
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """
    x_data = []
    for value in data.loc[:, data.columns != label_column].iterrows():
        new_value = []
        for input_column in input_columns:
            new_value = new_value + value[1][input_column]
        
        x_data.append(new_value)
    x_data = np.array(x_data)

    y_data = []
    for value in data[label_column]:
        y_data.append([int(value)])
    y_data = np.array(y_data)

    return x_data, y_data

def elaborate_data_10fold(data, train_index, test_index, iteration,
                          input_columns, label_column):
    """uses index obtained by 10Fold split and the iteration number to identify the validation partition
    return train, validation and test sets according to the selected label
    Args:
        data: dataframe
        train_index: index for the training set, according to 10Fold
        test_index: index fot the test set, according to 10Fold
        iteration: 10Fold's iteration number
        input_columns: list of columns of data to use as input for the model
        label_column: label column
    """

    """use last train fold as validation """
    if iteration == 0:
        i = 8
    else:
        i = iteration - 1

    a, b = list(KFold(n_splits=9).split(train_index))[i]
    val_index = train_index[b]
    train_index = train_index[a]

    x_train_GS, y_train_GS = elaborate_input(data.iloc[train_index, :], input_columns, label_column)
    x_val_GS, y_val_GS = elaborate_input(data.iloc[val_index, :], input_columns, label_column)

    x_test_GS, y_test_GS = elaborate_input(data.iloc[test_index, :], input_columns, label_column)

    return x_train_GS, y_train_GS, x_val_GS, y_val_GS, x_test_GS, y_test_GS


def get_tags_list():
    """ return the list of tags """
    with open(identity_tag_path, 'r') as fd:
        for line in fd:
            tag_list = line.lower().split('], ')[0].replace("'", "").strip('][').split(', ')
    return tag_list


def tag_embedding():
    """ Read tags list and compute their embeddings with USE 
    Return a dataframe with two columns:
        - tag: the tag name
        - tags_USE: embedding representation of the tag"""
    # read tags
    """
    with open(identity_tag_path, 'r') as fd:
        for line in fd:
            tmp = line.lower().split('], ')[0].replace("'", "").strip('][').split(', ')
    """
    #compute USE for each tag
    category_embedding = pd.DataFrame()
    category_embedding['tag'] = get_tags_list()
    category_embedding['tags_USE'] = use_preprocessing(category_embedding, 'tag')
    return category_embedding


def meme_tag_embedding(data, tag_column):
    """ Compute for each meme its embedding obtained through the mean of tags'embedding """
    category_embedding = tag_embedding()
    mean = []
    for index, row in data.iterrows():
        tags_emb = []
        for tag in data.loc[index,tag_column].split(' '):
            tags_emb.append(category_embedding.loc[category_embedding['tag'] == tag, 'tags_USE'].values[0])
        mean.append(np.mean(tags_emb, axis = 0).tolist())
    return mean

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) # numpy seed
    tf.random.set_seed(seed) # works for all devices (CPU and GPU)
