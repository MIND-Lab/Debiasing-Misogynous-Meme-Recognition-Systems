"""
Load identity terms and identity tags, verify its presence in synthetic data.
Il produces the file './Data/Syn_identity.csv' in which for every meme the presence of identity terms and tags is
indicated.
"""

from Utils import load_data, preprocessing

syn_df = load_data.load_syn_data()
syn_df['clear_text'] = preprocessing.apply_lemmatization_stanza(syn_df['Text Transcription'])

Identity_Terms = load_data.read_identity_terms()
Identity_Tags = load_data.read_identity_tags()
# Add labels indicating the term presence
preprocessing.add_subgroup_columns_from_text(syn_df, 'clear_text', Identity_Terms)

syn_df_clarifai = load_data.load_clarifai_data_syn()

if syn_df.file_name.to_list() == syn_df_clarifai.file_name.to_list():
    syn_df['clarifai'] = syn_df_clarifai['clarifai']
else:
    raise Exception('Check memes in clarifai_syn.csv')

"""Since a word can be both an identity tag and an identity term, i add 'Tag_' to denote when it's used as a term"""

syn_df['clarifai_elaborated'] = ''
for index, row in syn_df.iterrows():
    stringTag = ''
    for x in row.clarifai.split(' '):
        stringTag = stringTag + ' Tag_' + x
    syn_df.loc[index, 'clarifai_elaborated'] = stringTag

tmp = Identity_Tags

preprocessing.add_subgroup_columns_from_text(syn_df, 'clarifai_elaborated', Identity_Tags)
syn_df.to_csv('./Data/Syn_identity.csv', sep='\t', index=False)
