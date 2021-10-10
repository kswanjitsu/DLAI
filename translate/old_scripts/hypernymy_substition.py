# To run this script you have to install spaCy, scispaCy and the associated NER model for both called en_core_sci_sm
# Right now this script can receive a single test definition and use NER to identify the biomedical jargon in it
# It then searches against the file hypernymy_lookup table's 2nd column for the word phrase it matches w/ tfidf

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create function to use tfidf to search a list of wordphrases for a best match
def tfidf_search(query, list_to_search):
    # Get tf-idf matrix using fit_transform function
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list_to_search) # Store tf-idf representations of all docs
    query_vec = vectorizer.transform([query])  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
    results = cosine_similarity(X, query_vec).reshape((-1,))  # Op -- (n_docs,1) -- Cosine Sim with each doc
    print('####################################',query,'####################################')
    print(X.shape)  # (Number of wordphrases, Number of unique words)
    print('#################################################################################')

    for i in results.argsort()[-10:][::-1]:
        print(df.iloc[i, 0], "--", df.iloc[i, 1]) #, "--", df.iloc[i, 3])

# Import the dataframe that has the wordphrases to search and the hypernyms in a tree to search
columns = ['CUI','wordphrases','SNOMED_ID','hypernymy_tree']
df = pd.read_csv(
    "hypernymy_lookup_NSHT.txt",
    sep="|",
    names=columns
)

print(df)

# Convert the wordphrases we want to search to a list
wordphrases = df['wordphrases'].values.tolist()
wordphrases = [x.lower() for x in wordphrases]



"""
query = "suppressor cells"

# search the query against the wordphrase list using the created function above
tfidf_search(query, wordphrases)


test_text = 'Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC).'
import spacy
nlp = spacy.load("en_core_sci_sm")
doc = nlp(test_text)
for token in doc.ents:
    query = str(token).lower()
    tfidf_search(query, wordphrases)
"""
