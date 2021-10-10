

import pandas as pd
import numpy as np
import os 
import re
import operator
import nltk 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

columns = ['CUI','wordphrase','SNOMED_ID','SNOMED_concept','hypernymy_tree']

# This hypernymy_lookup file has the first concept in UMLS removed which is the "SNOMED"/"CUI" parent concept for all other concepts
# This is removed for obvious reasons. This is why the file has NSHT which means no snomed hypernymy tree

hypernymy_df = pd.read_csv(
    "hypernymy_lookup_NSHT.txt",
    sep="|",
    names=columns
)


## Create Vocabulary
vocabulary = set()
for doc in hypernyms:
    vocabulary.update(doc.split(','))

vocabulary = list(vocabulary)# Intializating the tfIdf model
tfidf = TfidfVectorizer(vocabulary=vocabulary)# Fit the TfIdf model
tfidf.fit(hypernyms)# Transform the TfIdf model
tfidf_tran=tfidf.transform(hypernyms)

def gen_vector_T(tokens):
    Q = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
    #print(tokens[0].split(','))
    for token in tokens[0].split(','):
        #print(token)
        try:
            ind = vocabulary.index(token)
            Q[ind]  = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def cosine_similarity_T(k, query):
    preprocessed_query = preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    d_cosines = []

    query_vector = gen_vector_T(q_df['q_clean'])
    for d in tfidf_tran.A:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    # print("")
    d_cosines.sort()
    a = pd.DataFrame()
    for i, index in enumerate(out):
        a.loc[i, 'index'] = str(index)
        a.loc[i, 'Subject'] = df_news['Subject'][index]
    for j, simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j, 'Score'] = simScore
    return a

cosine_similarity_T(10,'suppressor cells')

#test_text = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."
#replace_hypernyms(test_text, hypernymy_df)

