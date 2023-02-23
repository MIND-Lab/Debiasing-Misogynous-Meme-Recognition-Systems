# Debiasing Mysogynous Meme Recognition Systems

# Bias
This repository contains the code developed for "Recognizing Misogynous Memes: Biased Models and Tricky Archetypes". 
This repository contains several scripts to reproduce the results presented in the paper. The scripts allow both to extract image features (e.g. Clarifai tags and Azure captions) and to train the models and make predictions.
Two training approaches are proposed, with two different partitions of training data (8 train - 1 val - 1 test and 9 train - 1 val) and an additional one based on Bayesian Optimization.

___
## Installation
To run these scripts, all requirements must be met. Installation can be fulfilled manually or via `pip install -r requirements.txt`.

To install MMF run the following commands:
```
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .
```
For more information on installing MMF, as well as information on the requirements necessary for extracting features, consult the [MMF documentation](https://mmf.sh/).

## Prerequisites
Once the execution requirements are met, to execute the scripts it is necessary to arrange the data according to the following structure.
The dataset used in the realization of this project is the one proposed for the *Multimedia Automatic Misogyny Identification (MAMI)* Challenge at SemEval-2022 Task 5.

### Dataset Request Form
The datasets are exclusively reserved for the participants of SemEval-2022 Task 5 and are not to be used freely. The data may be distributed upon request and for academic purposes only. To request the datasets, please fill out the following form: https://forms.gle/AGWMiGicBHiQx4q98
After submitting the required info, participants will have a link to a folder containing the datasets in a zip format (trial, training and development) and the password to uncompress the files.

Data provided by the challenge have been enriched with a synthetic dataset. More information about the synthetic dataset creation can be found in the [paper](InProgress),
while information for download it can be found [here](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-).

## Directory Structure
The project should have the following structure:
```
├─ Data
|   ├─SYNTHETIC
|   ├─TEST
|   ├─TRAINING
|   ├─Categories.txt
|   ├─IdentityTerms.txt
|   ├─synthetic.csv
|   ├─test.xls
|   └─training.xls
├─ Utils
|   ├─ load_data.py
|   ├─ model_performances.py
|   └─ preprocessing.py
├─ ...
└─ requirements.txt
```
Some notes:
- The folders *TRAIN* and *TESTING* as long as the homonymous '.xls' files are provided from the MAMI challenge organizers.
- The folder *SYNTHETIC* and the relative '.csv' refers to synthetic data as described in the paper.
- 'Categories.txt' contains the list of Identity Tags (i.e. `['Animal', ...]`)
- 'IdentityTerms.txt' contains a list in which the first element is the list of misogynous Identity Terms, and the second element is the list of non-misogynous Identity Terms
    (i.e. `[['dishwasher',...],['memeshappen',...]]`)

___
## Running
All the scripts based on Azure require Azure captionings that can be obtained through `Azure_Captioning.py`; similarly, all clarifai-based scripts require clarifai tags which can be obtained through the `Clarifai_Tagging.py` script.

### Features Extractions
The above-mentioned scripts (**_`Azure_Captioning.py`_** and **_`Clarifai_Tagging.py`_**) regards image features extraction (i.e. Azure captions and Clarifai tags). 
Features are required to train the models and to make predictions.
All the images in the folders *TRAIN*, *TESTING* and *SYNTHETIC* that have 'PNG/JPG/JPEG' extensions will have their features extracted.
For the extraction of the features required by the transformers-based models (i.e. BERT and VisualBERT) execcute the files (**_`VisualBERT_1_FeaturesExtractions.py`_** and **_`VisualBERT_2_AnnotationsExtractions.py`_**).


#### **_`Azure_Captioning.py`_**
Extract image captions with Azure API. To be executed it requires (apart from the requirements listed in `requirements.txt`), Azure Cognitive services:
```pip install --upgrade azure-cognitiveservices-vision-computervision```
To extract Azure captions, please edit this file as follow:
- insert personal Azure key and endpoint (line 18 and 19)
- edit the path according to the data you want to process (i.e. training/test/synthetic) by selecting the associated paths (line 25 to 40).

The output is a 'tsv' file with image captions (also saved in a 'json' format).


#### **_`Clarifai_Tagging.py`_**
Extract image tags with Clarifai API. To be executed it requires (apart from the requirements listed in `requirements.txt`), Clarifai services:
```pip install clarifai-grpc```
To extract Clarifai tags, please edit this file as follows:
- insert personal key (line 145)
- edit the path according to the data you want to process (i.e. training/test/synthetic) by selecting the associated paths (line 21 to 40).

The output is a 'csv' file with image tags (according to 14 selected categories).


After the execution of these scripts, the Data folder should have the following structure:
```
─ Data
  ├─ Azure
  |   ├─azure_syn.tsv
  |   ├─azure_test.tsv
  |   └─azure_training.tsv
  ├─ Clarifai
  |   ├─clarifai_syn.csv
  |   ├─clarifai_test.csv
  |   └─clarifai_train.csv
  ├─SYNTHETIC
  ├─TEST
  ├─TRAINING
  ├─Categories.txt
  ├─IdentityTerms.txt
  ├─Syn_identity.csv
  ├─synthetic.csv
  ├─test.xls
  └─training.xls
```
Some notes:
-the `Syn_identity.csv` file define Candidate Bias Terms and Tags presence in synthetic data. Can be obtained through the execution of `CandidateBiasElementsOnSyn.py`.

### 10-Fold on Training data
The first approach is based on a 10Fold cross validation performed on training data (8 fold training, 1 fold validation, 1 fold test). This approach allows evaluating models performance on training data.
Scripts referring to this approach are identified by the label `10Fold`. (e.g. `BERT_1_TrainModels_10Fold.py` for BERT-text). Additional information about the procedure can be found in the paper.

### Candidate Bias Elements
Candidate Bias terms and tags are computed through a statistic-mathematical approach as shown in the paper.
Identity terms are computed via the script `CandidateBiasTerms.py`. This script compute a score for every term present in the training data such as the higher the score is, the more the term impacts the prediction in favor of misogynistic content.
Similarly, the lower (negative) the score, the more the term impacts the prediction in favor of non-misogynistic content. Terms close to zero are considered neutral. More details about the adopted approach can be found in the paper.

The script create a folder `./CandidateBiasTerms` with two files:
- `lemma_presence_stanza.csv`: a dataframe that represent the presence of every term in the dictionary in the mames in the
training data.
- `scores_Lemma_Stanza.csv`: the scores associated to every word in the dictionary

The 5 terms with the higher/lower score are selected as Candidate Bias Terms and saved in the file `./Data/IdentityTerms.txt`



