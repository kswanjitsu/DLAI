from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
#spacy.prefer_gpu()
import dask
dask.config.set(scheduler='threads')
import dask.dataframe as dd
nlp = spacy.load("en_core_sci_sm")

def replace_hypernyms(text,hypernymy_df):
    doc = nlp(text)
    new_text = ''
    for token in doc.ents:
        print(token)
        print(process.extractOne(str(token),hypernymy_df['wordphrase']))  # hypernymy_df['wordphrase']))

                #for line in hypernymy_df.itertuples():
                    #print(str(line[1]))
                    #print(token)
                   # print(process.extractOne(str(token), str(line[1])))# hypernymy_df['wordphrase']))


                #for line in hypernymy_df.itertuples():
                #    #print(line[1])
                #    #print(line[3])
                #    #print('\n')
                #    #print('\n')
                #    similar_words, _ = zip(*model.wv.most_similar(positive=[token.orth_]))

                    # Remove same lemma and words with underscore
                #    similar_words = [w for w in similar_words if '_' not in w and list(nlp(w))[0].lemma_ != token.lemma_]

                #    alt_word = similar_words[0] if len(similar_words) > 0 else token.orth_
                #    new_text += alt_word + ' '
                #else:
                #    new_text += token.orth_ + ' '

columns = ['CUI','wordphrase','SNOMED_ID','SNOMED_concept','hypernymy_tree']

# This hypernymy_lookup file has the first concept in UMLS removed which is the "SNOMED"/"CUI" parent concept for all other concepts
# This is removed for obvious reasons. This is why the file has NSHT which means no snomed hypernymy tree

hypernymy_df = dd.read_csv(
    "hypernymy_lookup_NSHT.txt",
    sample=50000000,
    sep="|",
    names=columns
)

test_text = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."
replace_hypernyms(test_text, hypernymy_df)

#print(hypernymy_df.compute())
#print(hypernymy_df['wordphrase'].compute())

