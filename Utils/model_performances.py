import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import json
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from keras.callbacks import EarlyStopping
from . import load_data
from sklearn.metrics import classification_report


SUBGROUP = 'subgroup'
SUBSET_SIZE = 'subset_size'
SUBGROUP_AUC = 'subgroup_auc'
NEGATIVE_CROSS_AUC = 'bpsn_auc'
POSITIVE_CROSS_AUC = 'bnsp_auc'

# ________________________________________________DEFINE MODEL__________________________________________________
# ____________________________________ Methods _________________________
activation_function_arr = ['sigmoid', 'relu', 'tanh', 'LeakyReLU']

def get_model(input_shape=512, activation_function='LeakyReLU', neurons=512, dropout=0.2, lr=0.00001, epsilon=1e-7):
    """ Builds the model with various parameters.
    Return the compiled model.

    :return: compiled model
    """

    model = Sequential()
    model.add(layers.Dense(neurons, input_shape=(input_shape,), activation=activation_function))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def get_trained_model(x_train, y_train, x_val, y_val,
    input_shape=512, activation_function='LeakyReLU', neurons=256, dropout=0.2, lr=0.00001, epsilon=1e-7, epochs=100,batch_size=64):
    """ A helper function first which builds the model with various parameters.
    The input parameters are used to train the model, while the 'iteration' value is used to
    detect the correct data to use on train.

    Args:
        iteration: 10Fold's iteration number
        dataset: dataframe with data
        train_index: index for the training set, according to 10Fold
        test_index: index fot the test set, according to 10Fold
        input_columns: list of columns of data to use as input for the model
        label_column: label column 
        input_shape: model's input shape
        lr: model optimizer's learning rate
        epsilon: model optimizer's epsilon value
        neurons: number of neuros for the model hidden layer (default set as input_shape/2)
        activation_function: model's activation function
        dropout: dropout value for the model's hidden layer
        epoch: epoch to use in training phase
        bathc_size: batch size

        :return: Trained model
    """
    if not isinstance(activation_function, str):
        activation_function=activation_function_arr[int(round(activation_function, 0))] #passaggio da float a int arrotondando
    #print(activation_function)
    
    # clear session
    tf.keras.backend.clear_session()

    

    model = get_model(input_shape,
                    activation_function=activation_function, 
                    neurons=neurons, 
                    dropout=dropout, 
                    lr=lr, 
                    epsilon=epsilon)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        verbose=0,
                        batch_size=batch_size,
                        callbacks=[es])

    return model, history


# ________________________________________________COMPUTE METRICS__________________________________________________
def compute_auc(y_true, y_pred):
    """
    :param y_true: list of real values
    :param y_pred: list of predicted values
    :return: Auc score
    """
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError as e:
        return np.nan


def model_family_auc(dataset, model_names, label_col):
    """
    Compute Auc score for every model in the model family.

    :param dataset: dataframe containing a column with real values (label_column) and a column for each model with the
        predicted labels
    :param model_names: list of models names (should be the same use for columns names)
    :param label_col: name of label column
    :return: dictionary with AUCs information
    """
    aucs = [
        compute_auc(dataset[label_col], dataset[model_name])
        for model_name in model_names
    ]
    return {
        'aucs': aucs,
        'mean': np.mean(aucs),
        'median': np.median(aucs),
        'std': np.std(aucs),
    }


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_negative_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])
    return compute_auc(examples[label], examples[model_name])


def compute_positive_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples])
    return compute_auc(examples[label], examples[model_name])


def calculate_overall_auc(df, model_name):
    true_labels = df['misogynous']
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def confusion_matrix_counts(df, score_col, label_col, threshold=0):
    """compute confusion rates _
    if threshold is not passed (=0), it computes matrix using predicted labels (from boolean values)"""
    if threshold:
        return {
            'tp': len(df[(df[score_col] >= threshold) & df[label_col]]),
            'tn': len(df[(df[score_col] < threshold) & ~(df[label_col])]),
            'fp': len(df[(df[score_col] >= threshold) & ~df[label_col]]),
            'fn': len(df[(df[score_col] < threshold) & df[label_col]]),
        }
    else:
        return {
            'tp': len(df[(df[score_col] == 1) & df[label_col]]),
            'tn': len(df[(df[score_col] == 0) & ~(df[label_col])]),
            'fp': len(df[(df[score_col] == 1) & ~df[label_col]]),
            'fn': len(df[(df[score_col] == 0) & df[label_col]]),
        }


# https://en.wikipedia.org/wiki/Confusion_matrix
def compute_confusion_rates(df, score_col, label_col, threshold=0):
    confusion = confusion_matrix_counts(df, score_col, label_col, threshold)

    actual_positives = confusion['tp'] + confusion['fn']
    actual_negatives = confusion['tn'] + confusion['fp']
    # True positive rate, sensitivity, recall.
    if actual_positives > 0:
        tpr = confusion['tp'] / actual_positives
    else:
        tpr = 0
    if actual_negatives > 0:
        # True negative rate, specificity.
        tnr = confusion['tn'] / actual_negatives
    else:
        tnr = 0

    # False positive rate, fall-out.
    fpr = 1 - tnr
    # False negative rate, miss rate.
    fnr = 1 - tpr
    if (confusion['tp'] + confusion['fp']) > 0:
        # Precision, positive predictive value.
        precision = confusion['tp'] / (confusion['tp'] + confusion['fp'])
    else:
        precision = 0

    accuracy = (confusion['tp'] + confusion['tn']) / (actual_positives + actual_negatives)
    if (precision + tpr) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * tpr) / (precision + tpr)
    auc = compute_auc(df[label_col], df[score_col])


    return {
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'precision': precision,
        'recall': tpr,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
    }


def compute_bias_metrics_for_subgroup_and_model(dataset,
                                                subgroup,
                                                model,
                                                label_col):
    """Computes per-subgroup metrics for one model and subgroup."""
    record = {
        SUBGROUP: subgroup,
        SUBSET_SIZE: len(dataset[dataset[subgroup]])
    }
    record[column_name(model, SUBGROUP_AUC)] = compute_subgroup_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, NEGATIVE_CROSS_AUC)] = compute_negative_cross_auc(
        dataset, subgroup, label_col, model)
    record[column_name(model, POSITIVE_CROSS_AUC)] = compute_positive_cross_auc(
        dataset, subgroup, label_col, model)
    return record


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        subgroup_record = compute_bias_metrics_for_subgroup_and_model(
            dataset, subgroup, model, label_col)
        # records.append(subgroup_record) #append function is deprecated
        records = [*records, subgroup_record]
    return pd.DataFrame(records)


def compute_bias_metrics_for_models(dataset,
                                    subgroups,
                                    models,
                                    label_col):
    """Computes per-subgroup metrics for all subgroups and a list of models."""
    output = None

    for model in models:
        model_results = compute_bias_metrics_for_model(dataset, subgroups, model,
                                                       label_col)
        if output is None:
            output = model_results
        else:
            output = output.merge(model_results, on=[SUBGROUP, SUBSET_SIZE])
    return output


# ____________________________________________Bias metrics_______________________________________________________________

def get_final_metric(bias_df, overall_auc_test, model_name):
    bias_score = np.average([
        bias_df[model_name + '_subgroup_auc'],
        bias_df[model_name + '_bpsn_auc'],
        bias_df[model_name + '_bnsp_auc']
    ])
    return np.mean([overall_auc_test, bias_score])


def compute_bias_score(bias_df_text, bias_df_image, model_name):
    bias_score_text = np.average([
        bias_df_text[model_name + '_subgroup_auc'],
        bias_df_text[model_name + '_bpsn_auc'],
        bias_df_text[model_name + '_bnsp_auc']
    ])
    bias_score_image = np.average([
        bias_df_image[model_name + '_subgroup_auc'],
        bias_df_image[model_name + '_bpsn_auc'],
        bias_df_image[model_name + '_bnsp_auc']
    ])
    return np.mean([bias_score_text, bias_score_image])


def get_final_multimodal_metric(bias_df_text, bias_df_image, overall_auc_test, model_name):
    """compute AUC Final Meme _ a metric bias proposed to compute multimodal bias in memes
    it considers bias in text and in image """
    bias_score = compute_bias_score(bias_df_text, bias_df_image, model_name)
    return np.mean([overall_auc_test, bias_score])

# added to compute bias score on new synthetic set
def compute_bias_score_nan(bias_df_text, bias_df_image, model_name):
    bias_score_text = np.nanmean([
        bias_df_text[model_name + '_subgroup_auc'],
        bias_df_text[model_name + '_bpsn_auc'],
        bias_df_text[model_name + '_bnsp_auc']
    ])
    bias_score_image = np.nanmean([
        bias_df_image[model_name + '_subgroup_auc'],
        bias_df_image[model_name + '_bpsn_auc'],
        bias_df_image[model_name + '_bnsp_auc']
    ])
    return np.mean([bias_score_text, bias_score_image])


def get_final_multimodal_metric_nan(bias_df_text, bias_df_image, overall_auc_test, model_name):
    """compute AUC Final Meme _ a metric bias proposed to compute multimodal bias in memes
    it considers bias in text and in image """
    bias_score = compute_bias_score_nan(bias_df_text, bias_df_image, model_name)
    return np.mean([overall_auc_test, bias_score])

# ____________________________________________PLOT________________________________________________________________
def plot_model_family_auc(dataset, model_names, label_col, min_auc=0.7, max_auc=1.0):
    result = model_family_auc(dataset, model_names, label_col)
    print('mean AUC:', result['mean'])
    print('median:', result['median'])
    print('stddev:', result['std'])
    plt.hist(result['aucs'])
    plt.gca().set_xlim([min_auc, max_auc])
    plt.show()
    return result


# ___________________________________________OTHER________________________________________________________

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def column_name(model, metric):
    return model + '_' + metric


def bias_metrics_on_file(txt_path, test_df, syn_df, identity_terms, model_names, label_column):
    """
    Compute metrics for bias-performance comparison and save them on file

    :param txt_path: path to txt file to store results
    :param test_df: dataframe containing predictions on test data (columns should correspond to Modelnames)
    :param syn_df: dataframe containing predictions on synthetic data (columns should correspond to Modelnames and to
        identity elements)
    :param identity_terms: list of Identity Terms
    :param model_names: list of model names
    :param label_column: true label column name
    :return: /
    """
    final_scores = {}
    bias_metrics = {}
    bias_value_metrics = {}
    overall_auc_metrics = {}

    for i in range(10):
        bias_metrics[i] = compute_bias_metrics_for_model(syn_df, identity_terms, model_names[i],
                                                         label_column)
        overall_auc_metrics[i] = calculate_overall_auc(test_df, model_names[i])
        final_scores[i] = get_final_metric(bias_metrics[i], overall_auc_metrics[i], model_names[i])

        bias_value_metrics[i] = np.average([
            bias_metrics[i][model_names[i] + '_subgroup_auc'],
            bias_metrics[i][model_names[i] + '_bpsn_auc'],
            bias_metrics[i][model_names[i] + '_bnsp_auc']
        ])

    file = open(txt_path, "a+")
    file.write('\n Bias _ AUC_Final\n')
    file.write('max overall auc (AUC raw): {max} \n'.format(
        max=max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys()))))
    file.write('Overall AUC values: {values}\n'.format(values=overall_auc_metrics))
    file.write('average overall auc (AUC raw): {mean} \n'.format(mean=np.average(list(overall_auc_metrics.values()))))

    file.write('max bias value: {max} \n'.format(max=max(zip(bias_value_metrics.values(), bias_value_metrics.keys()))))
    file.write('Bias values: {values}\n'.format(values=bias_value_metrics))
    file.write('average bias value: {mean} \n'.format(mean=np.average(list(bias_value_metrics.values()))))

    file.write('max AUC final: {max} \n'.format(max=max(zip(final_scores.values(), final_scores.keys()))))
    file.write('AUC final values: {values}\n'.format(values=final_scores))
    file.write('average AUC final: {mean} \n'.format(mean=np.average(list(final_scores.values()))))
    file.close()


def multilabel_bias_metrics_on_file(txt_path, test_df, syn_df, identity_terms, identity_tags, model_names,
                                    label_column):
    """
    Compute metrics for bias-performance comparison and save them on file

    :param txt_path: path to txt file to store results
    :param test_df: dataframe containing predictions on test data (columns should correspond to Modelnames)
    :param syn_df: dataframe containing predictions on synthetic data (columns should correspond to Modelnames and to
        identity elements)
    :param identity_terms: list of Identity Terms
    :param identity_tags: list of Identity Tags
    :param model_names: list of model names
    :param label_column: true label column name
    :return: /
    """

    final_multimodal_scores = {}
    bias_metrics_text = {}
    bias_metrics_image = {}
    bias_value_multimodal_metrics = {}
    overall_auc_metrics = {}

    for i in range(len(model_names)):
        bias_metrics_text[i] = compute_bias_metrics_for_model(syn_df, identity_terms, model_names[i],
                                                              label_column)
        bias_metrics_image[i] = compute_bias_metrics_for_model(syn_df, identity_tags, model_names[i],
                                                               label_column)
        overall_auc_metrics[i] = calculate_overall_auc(test_df, model_names[i])

        final_multimodal_scores[i] = get_final_multimodal_metric(bias_metrics_text[i],
                                                                 bias_metrics_image[i],
                                                                 overall_auc_metrics[i],
                                                                 model_names[i])

        bias_value_multimodal_metrics[i] = np.average([
            np.average([
                bias_metrics_text[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_text[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_text[i][model_names[i] + '_bnsp_auc']
            ]),
            np.average([
                bias_metrics_image[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_image[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_image[i][model_names[i] + '_bnsp_auc']
            ])
        ])

    file = open(txt_path, "a+")
    file.write('\n Bias _ AUC_Final_Multimodal\n')
    file.write(
        'max overall auc: {max} \n'.format(max=max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys()))))
    file.write('Overall AUC values: {values}\n'.format(values=overall_auc_metrics))
    file.write('average overall auc: {mean} \n'.format(mean=np.average(list(overall_auc_metrics.values()))))

    file.write('max multimodal bias: {max} \n'.format(
        max=max(zip(bias_value_multimodal_metrics.values(), bias_value_multimodal_metrics.keys()))))
    file.write('Multimodal bias values: {values}\n'.format(values=bias_value_multimodal_metrics))
    file.write(
        'average multimodal bias: {mean} \n'.format(mean=np.average(list(bias_value_multimodal_metrics.values()))))

    file.write('max AUC multimodal final: {max} \n'.format(
        max=max(zip(final_multimodal_scores.values(), final_multimodal_scores.keys()))))
    file.write('Multimodal final values: {values}\n'.format(values=final_multimodal_scores))
    file.write(
        'average AUC multimodal final: {mean} \n'.format(mean=np.average(list(final_multimodal_scores.values()))))
    file.close()

def multilabel_bias_metrics_on_file_BO(txt_path, test_df, syn_df, identity_terms, identity_tags, model_names,
                                    label_column):
    """
    Compute metrics for bias-performance comparison and save them on file. Deals with nan values generated by synthetic dataset

    :param txt_path: path to txt file to store results
    :param test_df: dataframe containing predictions on test data (columns should correspond to Modelnames)
    :param syn_df: dataframe containing predictions on synthetic data (columns should correspond to Modelnames and to
        identity elements)
    :param identity_terms: list of Identity Terms
    :param identity_tags: list of Identity Tags
    :param model_names: list of model names
    :param label_column: true label column name
    :return: /
    """

    final_multimodal_scores = {}
    bias_metrics_text = {}
    bias_metrics_image = {}
    bias_value_multimodal_metrics = {}
    overall_auc_metrics = {}

    for i in range(len(model_names)):
        bias_metrics_text[i] = compute_bias_metrics_for_model(syn_df, identity_terms, model_names[i],
                                                              label_column)
        bias_metrics_image[i] = compute_bias_metrics_for_model(syn_df, identity_tags, model_names[i],
                                                               label_column)
        overall_auc_metrics[i] = calculate_overall_auc(test_df, model_names[i])

        final_multimodal_scores[i] = get_final_multimodal_metric_nan(bias_metrics_text[i],
                                                                 bias_metrics_image[i],
                                                                 overall_auc_metrics[i],
                                                                 model_names[i])

        bias_value_multimodal_metrics[i] = np.nanmean([
            np.nanmean([
                bias_metrics_text[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_text[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_text[i][model_names[i] + '_bnsp_auc']
            ]),
            np.nanmean([
                bias_metrics_image[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_image[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_image[i][model_names[i] + '_bnsp_auc']
            ])
        ])

    file = open(txt_path, "a+")
    file.write('\n Bias _ AUC_Final_Multimodal\n')
    file.write(
        'max overall auc: {max} \n'.format(max=max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys()))))
    file.write('Overall AUC values: {values}\n'.format(values=overall_auc_metrics))
    file.write('average overall auc: {mean} \n'.format(mean=np.average(list(overall_auc_metrics.values()))))

    file.write('max multimodal bias: {max} \n'.format(
        max=max(zip(bias_value_multimodal_metrics.values(), bias_value_multimodal_metrics.keys()))))
    file.write('Multimodal bias values: {values}\n'.format(values=bias_value_multimodal_metrics))
    file.write(
        'average multimodal bias: {mean} \n'.format(mean=np.average(list(bias_value_multimodal_metrics.values()))))

    file.write('max AUC multimodal final: {max} \n'.format(
        max=max(zip(final_multimodal_scores.values(), final_multimodal_scores.keys()))))
    file.write('Multimodal final values: {values}\n'.format(values=final_multimodal_scores))
    file.write(
        'average AUC multimodal final: {mean} \n'.format(mean=np.average(list(final_multimodal_scores.values()))))
    file.close()



def confusion_rates_on_file(txt_path, data, model_names, label_column, threshold=0):
    """
    Compute mean value for the above-listed metrics for each model in model-names.

    :param txt_path: path to txt file to store results
    :param data: dataframe containing a column for each model with its predictions
    :param model_names: list of model names
    :param label_column: true label column name
    :param threshold: threshold value to use during predicted probability analysis. Default at 0.5
    :return: /
    """
    # Mean of score confusion_rates for models
    score = {'tpr': 0,
             'tnr': 0,
             'fpr': 0,
             'fnr': 0,
             'precision': 0,
             'recall': 0,
             'accuracy': 0,
             'f1': 0,
             'auc': 0,
             }

    real_values = np.array([])
    predict_values = np.array([])

    for model_name in model_names:
        predict_values = np.append(predict_values, data[model_name].tolist())
        real_values = np.append(real_values, data[label_column].tolist())

        score = dict(Counter(score) + Counter(compute_confusion_rates(data, model_name, label_column, threshold)))
    score = {key: value / len(model_names) for key, value in score.items()}
    """NB: the included measure of AUC is not accurate because it's performed on labels (therefore on thresholded data). 
    The follow row, correct that"""
    score['auc'] = model_family_auc(data, model_names, label_column)['mean']

    result_df = pd.DataFrame({'real': real_values.astype(int), 'pred': predict_values})
    file = open(txt_path, "a+")
    file.write('\n\n Test performances\n')
    file.write(json.dumps(score))
    file.write('\n') 
    file.write(classification_report(result_df['real'].values, (result_df['pred']>threshold).astype(int).values, target_names=['not_mis','mis']))
    file.write('\n AUC:') 
    file.write(str(compute_auc(result_df['real'].values, result_df['pred'].values)))
    file.close()


# _____________________________________________ Model Families ___________________________________________________
def model_family_name(model_names):
    """Given a list of model names, returns the common prefix."""
    prefix = os.path.commonprefix(model_names)
    if not prefix:
        raise ValueError("couldn't determine family name from model names")
    return prefix.strip('_')


def merge_family(model_family_results, models, metrics_list):
    output = model_family_results.copy()
    for metric in metrics_list:
        metric_columns = [column_name(model, metric) for model in models]
        output[column_name(model_family_name(models),
                           metric)] = output[metric_columns].values.tolist()
        output = output.drop(metric_columns, axis=1)
    return output


def compute_bias_metrics_for_model_families(dataset,
                                            subgroups,
                                            model_families,
                                            label_col):
    """Computes per-subgroup metrics for all subgroups and a list of model families (list of lists of models)."""
    output = None
    metrics_list = [SUBGROUP_AUC, NEGATIVE_CROSS_AUC, POSITIVE_CROSS_AUC]

    for model_family in model_families:
        model_family_results = compute_bias_metrics_for_models(
            dataset, subgroups, model_family, label_col)
        model_family_results = merge_family(model_family_results, model_family,
                                            metrics_list)
        if output is None:
            output = model_family_results
        else:
            output = output.merge(
                model_family_results, on=[SUBGROUP, SUBSET_SIZE])
    return output


def split_bias_metrics_columns(dataset, model_names, model_family_names):
    """create a columns for every execution splitting the one containing arrays
        dataset: obtained from tthe function 'compute_bias_metrics_for_model_families'
        model_names: list of model names
        model_family_names: name of the model family
    """
    sub_aucs = [el + "_subgroup_auc" for el in model_names]
    dataset[sub_aucs] = pd.DataFrame(dataset[model_family_names + '_v_subgroup_auc'].tolist(), index=dataset.index)

    bpsn_auc = [el + "_bpsn_auc" for el in model_names]
    dataset[bpsn_auc] = pd.DataFrame(dataset[model_family_names + '_v_bpsn_auc'].tolist(), index=dataset.index)

    bnsp_auc = [el + "_bnsp_auc" for el in model_names]
    dataset[bnsp_auc] = pd.DataFrame(dataset[model_family_names + '_v_bnsp_auc'].tolist(), index=dataset.index)
    return (dataset)


def split_aucs_columns(dataset, model_names, model_family_names):
    dataset[model_names] = pd.DataFrame(dataset[model_family_names].tolist(), index=dataset.index)
    return dataset


def get_model_family_final_multimodal_metric(bias_df_text, bias_df_image, overall_auc_test, model_names):
    model_family = model_family_name(model_names).split('_v')[0]

    bias_df_text = split_bias_metrics_columns(bias_df_text, model_names, model_family)
    bias_df_image = split_bias_metrics_columns(bias_df_image, model_names, model_family)
    overall_auc_test = split_aucs_columns(overall_auc_test, model_names, model_family)

    final_multimodal_metric = [
        get_final_multimodal_metric(bias_df_text, bias_df_image, overall_auc_test[model_name], model_name) for
        model_name in model_names]
    return {'model_family': model_family,
            'values': final_multimodal_metric,
            'mean': np.mean(final_multimodal_metric),
            'var': np.var(final_multimodal_metric),
            }


# ___________________________________________Bayesian Optimization______________________________________________________

def write_performance_on_file(filename, iteration, bias_metrics_text, bias_metrics_image, overall_auc_metrics,
                              final_multimodal_scores):
    with open(filename, 'a+') as f:
        f.write('___ITERATION {it}___\n'.format(it=iteration))
        f.write('bias_metrics_text: {metrica} \n'.format(metrica=bias_metrics_text))
        f.write('bias_metrics_image: {metrica} \n'.format(metrica=bias_metrics_image))
        f.write('overall_auc_metrics: {metrica} \n'.format(metrica=overall_auc_metrics))
        f.write('final_multimodal_scores:: {metrica} \n'.format(metrica=final_multimodal_scores))


def write_optimizer_on_file(filename, iteration, optimizer):
    with open(filename, 'a+') as f:
        f.write('___ITERATION {it}___\n'.format(it=iteration))
        for i, res in enumerate(optimizer.res):
            f.write("Iteration {}: \n\t{}".format(i, res))
        f.write("\n")

# ___________________________________________Performances on Syn 10Fold___________________________________________
def identity_element_presence(data, Identity_List, label_column):
  """return the elements present in the dataset in input.
  Each element in the input list is returned if in the given dataset there are both positive and negative examples for that element"""
  elements_present = []
  for element in Identity_List:
    if not data[data[element] & data[label_column]].empty and not data[data[element] & ~data[label_column]].empty:
      elements_present.append(element)
  return elements_present

def identity_element_presence_OR(data, Identity_List, label_column):
  """return the elements present in the dataset in input.
  Each element in the input list is returned if in the given dataset there are both positive and negative examples for that element"""
  elements_present = []
  for element in Identity_List:
    if not data[data[element] & data[label_column]].empty or not data[data[element] & ~data[label_column]].empty:
      elements_present.append(element)
  return elements_present

def model_family_auc_10Fold_Syn(dataset, model_names):
    """
    Compute Auc score for every model in the model family.

    :param dataset: dataframe containing, for each execution, a column with real values (label_column called 'label_'+modelname) 
        and a column for each model with the predicted labels (called model_name)
    :param model_names: list of models names (should be the same use for columns names)
    :return: dictionary with AUCs information
    """
    aucs = [
        compute_auc(dataset['label_'+model_name], dataset[model_name])
        for model_name in model_names
    ]
    return {
        'aucs': aucs,
        'mean': np.mean(aucs),
        'median': np.median(aucs),
        'std': np.std(aucs),
    }

def confusion_rates_on_file_10Fold_syn(txt_path, data, model_names, threshold=0):
    """
    Compute mean value for the above-listed metrics for each model in model-names.

    :param txt_path: path to txt file to store results
    :param data: dataframe containing, for each execution, a column with real values (label_column called 'label_'+modelname) 
        and a column for each model with the predicted labels (called model_name)
    :param model_names: list of model names
    :param threshold: threshold value to use during predicted probability analysis. Default at 0.5
    :return: /
    """
    # Mean of score confusion_rates for models
    score = {'tpr': 0,
             'tnr': 0,
             'fpr': 0,
             'fnr': 0,
             'precision': 0,
             'recall': 0,
             'accuracy': 0,
             'f1': 0,
             'auc': 0,
             }
    real_values = np.array([])
    predict_values = np.array([])

    for model_name in model_names:
        label_column = 'label_'+model_name
        predict_values = np.append(predict_values, data[model_name].tolist())
        real_values = np.append(real_values, data[label_column].tolist())
        
        
        score = dict(Counter(score) + Counter(compute_confusion_rates(data, model_name, label_column, threshold)))
    score = {key: value / len(model_names) for key, value in score.items()}
    """NB: the included measure of AUC is not accurate because it's performed on labels (therefore on thresholded data). 
    The follow row, correct that"""
    score['auc'] = model_family_auc_10Fold_Syn(data, model_names)['mean']
    result_df = pd.DataFrame({'real': real_values.astype(int), 'pred': predict_values})

    file = open(txt_path, "a+")
    file.write('\n\n Test performances 10_fold Split\n')
    file.write(json.dumps(score))
    file.write('\n') 
    file.write(classification_report(result_df['real'].values, (result_df['pred']>threshold).astype(int).values, target_names=['not_mis','mis']))
    file.write('\n AUC:') 
    file.write(str(compute_auc(result_df['real'].values, result_df['pred'].values)))
    file.close()

    # Performances on Syn 10Fold
def multimodal_bias_metrics_on_file_10Fold(txt_path, test_df, syn_df, identity_terms, identity_tags, model_names, label_column):
    """
    Compute metrics for bias-performance comparison and save them on file. Deals with nan values generated by synthetic dataset
    Allow to do deal with 10 fold execution on training data.

    :param txt_path: path to txt file to store results
    :param test_df: dataframe containing predictions on test data (columns should correspond to Modelnames)
    :param syn_df: dataframe containing predictions on synthetic data (columns should correspond to Modelnames and to
        identity elements)
    :param identity_terms: list of Identity Terms
    :param identity_tags: list of Identity Tags
    :param model_names: list of model names
    :return: /
    """

    final_multimodal_scores = {}
    bias_metrics_text = {}
    bias_metrics_image = {}
    bias_value_multimodal_metrics = {}
    overall_auc_metrics = {}
    syn_data=syn_df.copy()
    for i in range(len(model_names)):
        syn_df=syn_data
        syn_df['file_name'] = syn_df['file_name_'+model_names[i]]
        # add information about the presence of identity terms
        syn_df=syn_df.merge(load_data.load_syn_identity_data().drop(columns=['misogynous', 'Text Transcription']),
                        how='inner', on='file_name')
        syn_df['misogynous'] = syn_df['label_'+model_names[i]]
        # select the subset of identity elements that are present
        identity_terms_present = identity_element_presence(syn_df, identity_terms,label_column)
        tags_present = identity_element_presence_OR(syn_df, identity_tags, label_column)

        bias_metrics_text[i] = compute_bias_metrics_for_model(syn_df, identity_terms_present, model_names[i],
                                                              label_column)
        bias_metrics_image[i] = compute_bias_metrics_for_model(syn_df, tags_present, model_names[i],
                                                               label_column)
        overall_auc_metrics[i] = calculate_overall_auc(test_df, model_names[i])
        final_multimodal_scores[i] = get_final_multimodal_metric_nan(bias_metrics_text[i],
                                                                 bias_metrics_image[i],
                                                                 overall_auc_metrics[i],
                                                                 model_names[i])

        bias_value_multimodal_metrics[i] = np.nanmean([
            np.nanmean([
                bias_metrics_text[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_text[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_text[i][model_names[i] + '_bnsp_auc']
            ]),
            np.nanmean([
                bias_metrics_image[i][model_names[i] + '_subgroup_auc'],
                bias_metrics_image[i][model_names[i] + '_bpsn_auc'],
                bias_metrics_image[i][model_names[i] + '_bnsp_auc']
            ])
        ])

    file = open(txt_path, "a+")
    file.write('\n Bias _ AUC_Final_Multimodal_SYN 10fold Split\n')
    file.write(
        'max overall auc: {max} \n'.format(max=max(zip(overall_auc_metrics.values(), overall_auc_metrics.keys()))))
    file.write('Overall AUC values: {values}\n'.format(values=overall_auc_metrics))
    file.write('average overall auc: {mean} \n'.format(mean=np.average(list(overall_auc_metrics.values()))))

    file.write('max multimodal bias: {max} \n'.format(
        max=max(zip(bias_value_multimodal_metrics.values(), bias_value_multimodal_metrics.keys()))))
    file.write('Multimodal bias values: {values}\n'.format(values=bias_value_multimodal_metrics))
    file.write(
        'average multimodal bias: {mean} \n'.format(mean=np.average(list(bias_value_multimodal_metrics.values()))))

    file.write('max AUC multimodal final: {max} \n'.format(
        max=max(zip(final_multimodal_scores.values(), final_multimodal_scores.keys()))))
    file.write('Multimodal final values: {values}\n'.format(values=final_multimodal_scores))
    file.write(
        'average AUC multimodal final: {mean} \n'.format(mean=np.average(list(final_multimodal_scores.values()))))
    file.close()