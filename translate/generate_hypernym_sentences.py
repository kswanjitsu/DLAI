"""import spacy
from scispacy.hyponym_detector import HyponymDetector

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("hyponym_detector", last=True, config={"extended": False})


for asdf in open()
doc = nlp("Keystone plant species such as fig trees are good for the soil.")

i = str(doc1._.hearst_patterns)


if i != '[]':
    print(i)
"""

""""
import wikipedia as wiki
import pprint as pp

question = 'asdfasdfasdfasdfasdfrs'

results = wiki.search(question)

if str(results) != '[]':
    print(results)
    page = wiki.page(results[0])
    text = page.content
    print(text)
else:
    print("No result found")
"""

import dask.array as np

concepts = []
f = open("datasets/output/unique_umls_concept_strings.txt", "a+")
for line in open('./datasets/UMLS/MRCONSO.RRF', 'r'):
    linesplit = line.split('|')
    source_lang = linesplit[1]
    if source_lang == 'ENG':
        concept = linesplit[14]
        #print(concept)
        concepts.append(concept)

a = np.array(concepts)
u, indices = np.unique(a, return_index=True)
print(u)




