# to run this script you have to install spaCy, scispaCy and the associated NER model for both called en_core_sci_sm
# right now this script can receive a single test definition and use NER to identify the biomedical jargon in it
# it then searches against the file hypernymy_lookup table's 2nd column for the word phrase it matches w/ tfidf
# next to come is selecting the top "hit" from the search and then using the 4th columns hypernymy tree to substitute

"""It would be cool to first try to find the token in wordnet, get hypernyms from there and if not found then search
the UMLS corpus -- rationale in paper is that wordnet is comprehensive, UMLS is_a relationship are poor substitutes for
true hypernyms, which wordnet has - need to remake figure and rewrite a bit if you do this"""

"""Maybe as corpus grows larger, bin each lookupfile alphabetically, can parallelize functions and target the lookup file
by starting letter, would need a lot of if then statements and this would have to occur after the NER model identifies 
biomedical jargon"""

"""Maybe regular ole spacy tokens is better w/ wordnet, alternatively tokenize w/ scibert and search wordnet for 
each word, not word phrase for hypernyms"""

# Beware per Josh - can run through multiple languages - some of the scoring systems have better validity - Holly ? ascessibility engineer facebook
import sys
sys.path.insert(0, '../sequence-labeler-master')
from complex_labeller import Complexity_labeller
model_path = '../cwi_seq.model'
temp_path = '../temp_file.txt'

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##################################################################################
# create function to use tfidf to search a list of wordphrases for a best match
def tfidf_search(query, list_to_search):
    # Get tf-idf matrix using fit_transform function
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list_to_search) # Store tf-idf representations of all docs
    query_vec = vectorizer.transform([query])  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
    results = cosine_similarity(X, query_vec).reshape((-1,))  # Op -- (n_docs,1) -- Cosine Sim with each doc
    print("#####################################",query,"#####################################")
    print(X.shape) # (Number of wordphrases, Number of unique words)
    print("###################################################################################")
    for i in results.argsort()[-1:][::-1]:
        print(df.iloc[i, 0], "--", df.iloc[i, 1],"--", df.iloc[i, 3])

########################################################################

# import the dataframe that has the wordphrases to search and the hypernyms in a tree to search
columns = ['CUI','wordphrases','SNOMED_ID','hypernymy_tree']
df = pd.read_csv(
    "hypernymy_lookup.txt",
    sep="|",
    names=columns
)


# convert the wordphrases we want to search to a list
wordphrases = df['wordphrases'].values.tolist()
wordphrases = [x.lower() for x in wordphrases]

#################################################################################
#import spacy
#nlp_engsm = spacy.load('')

import readability_v_ks
nlp_scibert = readability_v_ks.spacy.load("en_ner_bc5cdr_md")
nlp_scibert.add_pipe('readability')

#from scispacy.abbreviation import AbbreviationDetector
#nlp_scibert.add_pipe("abbreviation_detector")
#test_text = 'Atrial fibrillation (also called AFib or AF) is a quivering or irregular heartbeat (arrhythmia) that can lead to blood clots, stroke, heart failure and other heart-related complications. At least 2.7 million Americans are living with AFib.'

test_text = 'A squamous cell carcinoma of the bladder arising from metaplastic epithelium. It represents less than 10% of bladder carcinomas. The exception is the Middle East along the Nile Valley, where it represents the most common form of carcinoma because of the endemic nature of schistosomiasis. Bladder squamous cell carcinoma is often associated with long-standing chronic inflammation of the bladder and usually has a poor prognosis. The diagnosis of squamous cell carcinoma of the bladder should be reserved for those tumors that are predominantly keratin forming. Check for active clinical trials using this agent. "'
doc_med = nlp_scibert(test_text)
#doc_eng = nlp_engsm(test_text)
print(doc_med._.flesch_kincaid_grade_level)
print(doc_med._.flesch_kincaid_reading_ease)
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

"""nlp_wordnet.add_pipe("spacy_wordnet")"""
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn


# do this for every scispacy nlp and store the token / token types
print([(X.text, X.tag_, X.ent_type_) for Y in doc_med.noun_chunks for X in Y])
#for token in doc_med.ents:
    #print(token.ent_type_)

    #query = str(token).lower()
    #tfidf_search(query, wordphrases)



    #token_string = (str(token))
    #token2 = wn.synsets(token_string)
    #print('\n')
    #print('token: ', token)
    #print('synset: ', token2)

    #hypernyms = []
    #for i in token2:
    #    synonym = str(i).split("'")[1]
    #    hypernym = i.hypernyms()
    #    for j in hypernym:
    #        hypernyms.append(j)
    #print("synonym's hypernyms: ", hypernyms)


#left here where we are pulling hypernyms from synsets - going to need to figure out which hypernym to pull had the thought that if noun can pull it, if noun 1 = noun 2 pass == or pull all select for the lowest FK score
#use third / all scispacy and spacy if needed
# restructure the for loop above as functions to call and then can parallelize with dask, then if sets for types are empty use tfidf
