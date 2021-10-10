import spacy
import scispacy
spacy.require_gpu()
from scispacy.abbreviation import AbbreviationDetector
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("abbreviation_detector")

def display_entities(nlp,document):
	doc = nlp(document)
	displacy_image = displacy.render(doc, jupyter=True, style='ent')
	entity_and_label = pprint(set([(X.text, X.label_) for X in doc.ents]))
	return displacy_image, entity_and_label

def show_medical_abbreviations(nlp, document):
	doc = nlp(document)
	abbreviated = list(set([f"{abrv}  {abrv._.long_form}" for abrv in doc._.abbreviations]))
	return abbreviated



for line in open('../datasets/output/dataset_parsed_all_mimic_test_notes.txt', 'r'):
	test_doc = line
	print(show_medical_abbreviations(nlp, test_doc))
