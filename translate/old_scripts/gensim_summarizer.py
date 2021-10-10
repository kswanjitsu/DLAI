# use gensim to summarize then use scispacy to find words then build hypernymy / synonymy substition with cuDF hypernymy tree

import gensim
import spacy
spacy.prefer_gpu()

nlp = spacy.load("en_core_sci_sm")

text = """
Myeloid derived suppressor cells (MDSC) are immature 
myeloid cells with immunosuppressive activity. 
They accumulate in tumor-bearing mice and humans 
with different types of cancer, including hepatocellular 
carcinoma (HCC).
"""

embeddings_file = 'word_embeddings/word2vec/GoogleNews-vectors-negative300.bin'
model = gensim.models.Word2Vec.load("GoogleNews-vectors-negative300.bin.gz")


#doc = nlp(text)
#doc = nlp(text)
#print(doc.ents)


def replace_nouns(orig_text):
    doc = nlp(orig_text)
    new_text = ''
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ['NOUN', 'PROPN'] and token.orth_ in model:
                similar_words, _ = zip(*model.wv.most_similar(positive=[token.orth_]))

                # Remove same lemma and words with underscore
                similar_words = [w for w in similar_words if '_' not in w and list(nlp(w))[0].lemma_ != token.lemma_]

                alt_word = similar_words[0] if len(similar_words) > 0 else token.orth_
                new_text += alt_word + ' '
            else:
                new_text += token.orth_ + ' '

    return new_text

replace_nouns(text)
print(new_text)
