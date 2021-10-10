import spacy

spacy.require_gpu(0)
nlp = spacy.load("en_core_web_lg")
for key in list(nlp.vocab.vectors.key2row):
    try:
        word = nlp.vocab.strings[key]
    except KeyError:
        del nlp.vocab.vectors.key2row[key]
nlp.to_disk("/mod_en_core_web_lg")
