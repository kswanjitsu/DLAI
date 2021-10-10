import sys
sys.path.insert(0, '')
from complex_labeller import Complexity_labeller

model_path = '../cwi_seq.model'
temp_path = '../temp_file.txt'

model = Complexity_labeller(model_path, temp_path)

import readability_v_ks

nlp = readability_v_ks.spacy.load("en_ner_bc5cdr_md")
nlp.add_pipe('readability')

from nltk.corpus import wordnet as wn


def automatic_translation(document):

    # Converting example sentence/document
    test_document = document
    new_document = test_document
    Complexity_labeller.convert_format_string(model, test_document)

    # %%

    # The `get_dataframe` method returns a dataframe containing the original tokenized sentence, binary complexity labels and complex class probabilities.
    # If a word recieves a binary label = 1, it has been classified as a complex word.
    dataframe = Complexity_labeller.get_dataframe(model)

    # Access binary labeling information from the dataframe format:
    cw_list = list(zip(dataframe['sentences'].values[0], dataframe['labels'].values[0], dataframe['probs'].values[0]))

    # get_bin_labels returns the binary complexity labels for the input
    # bin_label_list = Complexity_labeller.get_bin_labels(model)

    # The `get_prob_labels` method returns the probability of each token belonging to the complex class.
    # prob_label_list = Complexity_labeller.get_prob_labels(model)




    doc_med = nlp(test_document)



    for token in doc_med.ents:
        token_string = str(token)
        token_string_list = token_string.split(' ')
        #print('\n')
        #print('token: ', token)

        for i in cw_list:
            cw = i[0]
            cw_bin_complexity = i[1]
            cw_prob_complexity_1 = i[2]
            cw_prob_complexity_2 = str(cw_prob_complexity_1[1])
            cw_prob_complexity_3 = float(cw_prob_complexity_2)
            # print(cw_bin_complexity_3.)

            if cw in token_string_list and cw_bin_complexity == 1 and cw_prob_complexity_3 >= 0.5:
                #print(cw)
                token2 = wn.synsets(cw)
                try:
                    #print(token2[0])
                    hypernym = token2[0].hypernyms()
                    hypernym = str(hypernym).split("'")[1].split(".")[0]
                    #print(hypernym)
                    new_document = new_document.replace(cw, hypernym)
                    #print(new_document)
                    #print('-------')
                except:
                    pass
                    #print('no synset')
                    #print('-------')


    og_score = doc_med._.flesch_kincaid_grade_level
    og_grade = doc_med._.flesch_kincaid_reading_ease

    new_document = new_document.replace('_',' ')

    doc_new = nlp(new_document)
    new_score = doc_new._.flesch_kincaid_reading_ease
    new_grade = doc_new._.flesch_kincaid_grade_level

    if new_score <= 8 and new_grade > og_score: # I know this seems backwards but due to some weird shit with word embeddings the matrices I think are like run backwards, so it switches the axes, if you put in right order by variable name, the names will be right but the numbers will be wrong.
        print('\n')
        print(test_document)
        print(og_grade, og_score)

        print(new_document)
        print(new_score, new_grade)

    # %%

for line in open("/home/karl/PycharmProjects/pythonProject/UMLS/MRDEF.RRF", 'r'):
    source = str(line.split('|')[4])
    #print(source)
    if source == 'MSH':
        document = str(line.split('|')[5])
        automatic_translation(document.lower())
    else:
        pass
