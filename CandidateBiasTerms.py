"""
Probabilistic approach to identify identity terms from Training data.
This approach is based on Spacy Stanza lemmatization in order to group concepts; apply several preprocessing
techniques such as lower case conversion and special character removal. It considers only words that appear at least
10 times in training data.

Requires Spacy stanza: "pip install spacy-stanza"
"""

import pandas as pd
import string
from collections import Counter
import stanza
import spacy_stanza
import re
from Utils import load_data
import os
from tqdm import tqdm

# _______________________________________________UTILS_______________________________________________________
stopwords = ["a", "about", "above", "above", "across", "afterwards", "again", "against",
             "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
             "amoungst",
             "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
             "around",
             "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "beforehand",
             "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but",
             "by",
             "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "describe", "detail", "do", "done",
             "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
             "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify",
             "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from",
             "front",
             "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "hence", "here", "hereafter",
             "hereby",
             "herein", "hereupon", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into",
             "is",
             "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "meanwhile", "might",
             "mill",
             "more", "moreover", "most", "mostly", "move", "much", "must", "name", "namely", "neither", "never",
             "nevertheless",
             "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "now", "nowhere", "of", "off", "often",
             "on", "once",
             "one", "only", "onto", "or", "other", "others", "otherwise", "out", "over", "part", "per", "perhaps",
             "please",
             "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "should",
             "show",
             "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "sometime", "sometimes",
             "somewhere",
             "still", "such", "system", "take", "ten", "than", "that", "the", "then", "thence", "there", "thereafter",
             "thereby",
             "therefore", "therein", "thereupon", "these", "thick", "thin", "third", "this", "those", "though", "three",
             "through",
             "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
             "two", "un",
             "under", "until", "up", "upon", "very", "via", "was", "well", "were", "what", "whatever", "when", "whence",
             "whenever",
             "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
             "while", "whither",
             "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
             "the",
             "ve", "re", "ll", "10", "11", "18", "oh", "s", "t", "m", "did", "don", "got"]


def clear_text_lemma(testo):
    """
    Remove punctuation, brings to lowercase, remove special char, apply Stanza lemmatization

    :param testo: text to process
    :return: processed text
    """
    rev = []

    testo = testo.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    testo = testo.lower()
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub('[^A-Za-z0-9 ]+', '', testo)
    testo = " ".join(testo.split())  # single_spaces

    doc = nlp(testo)
    for token in doc:
        rev.append(token.lemma_)

    for word in list(rev):  # iterating on a copy since removing will mess things up
        if word in stopwords:
            rev.remove(word)
    return rev


def frequent_words(text, n):
    """
    remove from the dictionary words that appears less than n times.
    :param text: text to process
    :param n: minimum occurrences number
    """
    split_it = []
    for row in text:
        split_it.extend(row.split())

    counter = Counter(split_it)

    frequent_words = []
    for x in counter.most_common():
        if x[1] >= n:
            frequent_words.append(x[0])
        else:
            print(x)

    return frequent_words


epsilon = 5000

if not os.path.exists('./IdentityTerms/'):
    os.makedirs('./IdentityTerms/')

# ____________________________________________________Laod Data______________________________________________
stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

data = load_data.load_training_data()
data['clear text'] = ''

# _____________________________________________Dictionary ________________________________________________
# dictionary includes words that appear at least 10 times

dictionary = []

for index, row in tqdm(data.iterrows()):
    data.loc[index, 'clear text'] = str(clear_text_lemma(row[2])).replace("'", '').replace(",", '').replace("[",
                                                                                                            '').replace(
        "]", '').replace("\"", '')

dictionary = frequent_words(data['clear text'], 10)

# Word dataframe: a column for each word in the dictionary, with a boolean value to represent its presence in the meme
word = pd.DataFrame(columns=['file_name', 'misogynous'] + dictionary)

for index, row in tqdm(data.iterrows()):
    new_line = list(data.loc[index, ['file_name', 'misogynous']])
    word = word.append({'file_name': data.loc[index, 'file_name'],
                        'misogynous': data.loc[index, 'misogynous']}, ignore_index=True)
    for w in row['clear text'].split():
        if w in dictionary:
            # print(w)
            word.loc[word['file_name'] == data.loc[index, 'file_name'], w] = 1

word.to_csv('./IdentityTerms/lemma_presence_stanza.csv')

word = word.fillna(0)

# ____________________________________ Conditional probabilities _____________________________________________
col = ['class_misogynous']
col.extend(word.columns[2:len(word.columns)].tolist())

condizionate = pd.DataFrame(columns=col)
condizionate.loc[0, 'class_misogynous'] = 'misogynous'
for x in word.columns[2:len(word.columns)]:
    if len(word.loc[word['misogynous'] == 1, x].value_counts()) == 2:
        condizionate.loc[0, x] = word.loc[word['misogynous'] == 1, x].value_counts()[1] / \
                                 word.loc[word['misogynous'] == 1, x].shape[0]
    elif 1 in word.loc[word['misogynous'] == 1, x].tolist():
        condizionate.loc[0, x] = 1 - (1 / epsilon)
    else:
        condizionate.loc[0, x] = (1 / epsilon)

condizionate.loc[1, 'class_misogynous'] = '¬misogynous'
for x in word.columns[2:len(word.columns)]:
    if len(word.loc[word['misogynous'] == 0, x].value_counts()) == 2:
        condizionate.loc[1, x] = word.loc[word['misogynous'] == 0, x].value_counts()[1] / \
                                 word.loc[word['misogynous'] == 0, x].shape[0]
    elif 1 in word.loc[word['misogynous'] == 0, x].tolist():
        condizionate.loc[1, x] = 1 - (1 / epsilon)
    else:
        condizionate.loc[1, x] = (1 / epsilon)

# ___________________________________P(M|tags)__________________________________________________
calcolate = pd.DataFrame(columns=['meme', 'eq', 'valore'])

for index, row in word.iterrows():
    print('\n')

    tags = []
    eq = 'P(M|'
    for i in range(2, len(word.columns)):
        if row[i] == 1:
            tags.append(word.columns[i])
            eq = eq + word.columns[i] + ' '
    eq = eq + ')'
    print(eq)

    # values to be normalized
    value_pos = 0.5
    value_neg = 0.5
    conto = '0.5'

    for x in tags:
        conto = conto + '*' + str(condizionate.loc[0, x])
        value_pos = value_pos * condizionate.loc[0, x]
        value_neg = value_neg * condizionate.loc[1, x]

    # Normalization
    somma = value_pos + value_neg
    value_pos = value_pos / somma
    value_neg = value_neg / somma

    calcolate = calcolate.append(pd.DataFrame({'meme': [index + 1], 'eq': [eq], 'valore': [value_pos]}))
    print(value_pos)
    result = value_pos

    eq = 'P(¬M|'
    for i in tags:
        eq = eq + i + ' '
    eq = eq + ')'
    print(eq)
    print(value_neg)

# ______________________________________Remove tags P(M|tags-{tag})___________________________________________
rimozioneTag = pd.DataFrame(columns=['meme', 'tagTolto', 'eq', 'valore'])

for index, row in word.iterrows():
    print('\n')
    tags = []

    for i in range(2, len(word.columns)):
        if row[i] == 1:
            tags.append(word.columns[i])

    # compute probability without selected tag
    for tag in tags:
        tmp = tags.copy()
        tmp.remove(tag)
        eq = 'P(M|'

        value_pos = 0.5
        value_neg = 0.5
        conto = '0.5'

        # values to normaize
        for x in tmp:
            eq = eq + x + ' '
            conto = conto + '*' + str(condizionate.loc[0, x])
            value_pos = value_pos * condizionate.loc[0, x]
            value_neg = value_neg * condizionate.loc[1, x]

        eq = eq + ')'
        print(eq)
        print(conto)

        # Normalization
        somma = value_pos + value_neg
        value_pos = value_pos / somma
        value_neg = value_neg / somma
        print(value_pos)

        rimozioneTag = rimozioneTag.append(
            pd.DataFrame({'meme': [index + 1], 'tagTolto': tag, 'eq': [eq], 'valore': [value_pos]}))

rimozioneTag = rimozioneTag.reset_index(drop=True)

# ________________________________________ Meme scores___________________________________________________
# valMeme-value
rimozioneTag['score'] = 0
for index, row in rimozioneTag.iterrows():
    rimozioneTag.loc[index, 'score'] = calcolate.loc[calcolate['meme'] == row[0], 'valore'].values[0] - row[3]

# Compute mean per tag and save in dataframe
scores_df = pd.DataFrame(columns=['word', 'score'])
for tag in word.columns[2:len(word.columns)]:
    media = sum(rimozioneTag.loc[rimozioneTag['tagTolto'] == tag, 'score'].tolist()) / len(
        rimozioneTag.loc[rimozioneTag['tagTolto'] == tag, 'score'].tolist())
    scores_df = scores_df.append({'word': tag, 'score': media}, ignore_index=True)

scores_df = scores_df.sort_values(by=['score'], ascending=False)
scores_df.to_csv('./IdentityTerms/scores_Lemma_Stanza.csv', index=False)

# _______________________________Score analysis___________________________________
# Remove words with less than 2 char
short = []
for w in scores_df.word:
    if len(w) <= 2:
        short.append(scores_df[scores_df['word'] == w].index[0])
scores_df = scores_df.drop(index=short)

# first/last 10 terms
identity_misogynous = scores_df[0:5].word.tolist()
identity_non_misogynous = scores_df[scores_df.shape[0] - 5:scores_df.shape[0]].word.tolist()
identity_non_misogynous.reverse()

identity_terms = [identity_misogynous, identity_non_misogynous]

with open('./Data/IdentityTerms.txt', 'w') as f:
    f.write(str(identity_terms))
