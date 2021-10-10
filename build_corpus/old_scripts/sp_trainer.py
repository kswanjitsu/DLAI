import pandas as pd
mrrel_columns = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL','RG','DIR','SUPPRESS','CVF']

mrrel = pd.read_csv(
    "s3://sagemaker-studio-757570088617-lvhv3fp3v5/MRREL.RRF",
    header=None,
    sep="|",
    #lineterminator="\n",
    names=mrrel_columns
)

print(type(mrrel))
print(mrrel)

import spacy
spacy.prefer_gpu()
from spacy.matcher import Matcher
import os





patterns = [
[{"POS": "NOUN"}, {"TEXT": "which is an example of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which is a kind of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which is a class of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "an example of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "a kind of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "a class of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"},  {"TEXT": "some"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "any"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "any"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "some"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which is called"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "is"}, {"POS": "ADJ"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "is a"}, {"IS_ALPHA": True}, {"TEXT": "case of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "is a case of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "is a"}, {"POS": "NOUN"}, {"TEXT": "that"}],
[{"POS": "NOUN"}, {"TEXT": "is an"}, {"POS": "NOUN"}, {"TEXT": "that"}],
[{"TEXT": "properties of"}, {"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}],
[{"TEXT": "properties of"}, {"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
[{"TEXT": "features of"},  {"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}],
[{"TEXT": "features of"},  {"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
[{"TEXT": "unlike"}, {"TEXT":"most"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "unlike"}, {"TEXT":"all"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "unlike"}, {"TEXT":"any"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "unlike"}, {"TEXT":"other"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "like"}, {"TEXT": "most"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "like"}, {"TEXT":"all"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "like"}, {"TEXT":"any"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"TEXT": "like"}, {"TEXT":"other"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT":"including"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT":"including"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT":"including"}, {"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "such as"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"TEXT": "such as"}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such as"}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"TEXT": "such as"}, {"POS": "NOUN"}, {"TEXT": "or"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "include"},{"POS": "NOUN"},{"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "include"}, {"POS": "NOUN"},{"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "include"},{"POS": "NOUN"},{"IS_PUNCT": True},{"TEXT": "and"}],	
[{"POS": "NOUN"}, {"TEXT": "include"}, {"POS": "NOUN"},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "include"},{"POS": "NOUN"},{"IS_PUNCT": True},{"TEXT": "or"}],	
[{"POS": "NOUN"}, {"TEXT": "include"}, {"POS": "NOUN"},{"TEXT": "or"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "especially"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and "}, {"TEXT": "any"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and "}, {"TEXT": "any"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and "}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and "}, {"TEXT": "any"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "some"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "some"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "some"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and be a"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and be a"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "like"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "like"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "like"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "like"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "like"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "like"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "like"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "like"}, {"POS": "NOUN"}, {"TEXT": "or"}], 
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "as"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"TEXT": "as"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"TEXT": "such"}, {"POS": "NOUN"}, {"TEXT": "as"}, {"POS": "NOUN"}, {"TEXT": "or"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "like"}, {"TEXT": "other"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "like"}, {"TEXT": "other"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "the"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "the"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "these"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "these"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "these"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "one"}, {"TEXT": "of"}, {"TEXT": "these"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "be"}, {"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "be"}, {"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"TEXT": "example"}, {"TEXT": "of"}, {"POS": "NOUN"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "or"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "for"}, {"TEXT": "example"}, {"IS_PUNCT": True}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "for"}, {"TEXT": "example"}],
[{"POS": "NOUN"}, {"TEXT": "for"}, {"TEXT": "example"}, {"IS_PUNCT": True}],
[{"POS": "NOUN"}, {"TEXT": "for"}, {"TEXT": "example"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "call"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "call"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "call"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "call"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "name"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "name"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "name"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "which"}, {"TEXT": "be"}, {"TEXT": "name"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "mainly"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "mostly"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "notably"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "particularly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "particularly"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "particularly"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "particularly"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "principally"}, {"POS": "NOUN"}, {"IS_PUNCT": True}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "principally"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "principally"}, {"POS": "NOUN"}, {"IS_PUNCT": True}],
[{"POS": "NOUN"}, {"TEXT": "principally"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "in particular"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "except"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "except"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "except"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "except"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "except"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "except"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "except"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "except"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "other than"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "e.g."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"IS_PUNCT": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"IS_PUNCT": True},{"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "i.e."}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "a"}, {"TEXT": "kind"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "a"}, {"TEXT": "kind"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "a"},{"TEXT": "kind"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "a"},{"TEXT": "kind"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "a"}, {"TEXT": "form"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "a"}, {"TEXT": "form"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "a"},{"TEXT": "form"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "a"},{"TEXT": "for"}, {"TEXT": "of"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "looks"},{"TEXT": "like"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "looks"},{"TEXT": "like"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "looks"},{"TEXT": "like"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "looks"},{"TEXT": "like"}, {"POS": "NOUN"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "sounds"},{"TEXT": "like"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "sounds"},{"TEXT": "like"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "sounds"},{"TEXT": "like"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "sounds"},{"TEXT": "like"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "be"},{"TEXT": "similar"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"TEXT": "ir"}],
[{"POS": "NOUN"}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "example"}, {"TEXT": "of"},{"TEXT": "this"}, {"TEXT": "be"},{"POS": "NOUN"}, {"TEXT": "ir"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "type"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "type"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "type"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "type"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "type"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "type"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "type"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "type"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"},  {"POS": "NOUN"}, {"TEXT": "type"}],
[{"POS": "NOUN"}, {"TEXT": "and"},  {"POS": "NOUN"}, {"TEXT": "type"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"},  {"POS": "NOUN"}, {"TEXT": "type"}],
[{"POS": "NOUN"}, {"TEXT": "or"},  {"POS": "NOUN"}, {"TEXT": "type"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "whether"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"TEXT": "compare"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "with"}, {"POS": "NOUN"}],
[{"TEXT": "compare"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "with"}, {"POS": "NOUN"}],
[{"TEXT": "compare"}, {"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "with"}, {"POS": "NOUN"}],
[{"TEXT": "compare"}, {"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "with"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"},  {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"},  {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"},  {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "compare"}, {"TEXT": "to"}, {"POS": "NOUN"},  {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "among"}, {"IS_ALPHA": True}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "as"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "as"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "as"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "as"}, {"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "for instance"}], 
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True},  {"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "for instance"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}, {"TEXT": "sort"}, {"TEXT": "of"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}, {"TEXT": "sort"}, {"TEXT": "of"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "and"}, {"TEXT": "sort"}, {"TEXT": "of"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"TEXT": "or"}, {"TEXT": "sort"}, {"TEXT": "of"},{"POS": "NOUN"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "and"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"IS_PUNCT": True}, {"TEXT": "or"}],
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"TEXT": "and"}], 
[{"POS": "NOUN"}, {"TEXT": "which"}, {"TEXT": "may"}, {"TEXT": "include"}, {"POS": "NOUN"}, {"TEXT": "or"}],

  ]


with open("./final_hypernym_output.txt", "w") as new_file:
	new_file.write("Example text")

for file in os.listdir("./wiki_output/"):
	try:
		if file.endswith(".txt"):
			filename = os.path.join("./wiki_output/", file)
			with open(filename) as f:
				nlp = spacy.load("en_core_web_trf")
				matcher = Matcher(nlp.vocab)
				nlp.max_length = 200000
				TEXTS = f.read()
				#print(TEXTS[0:100])
				matcher.add("HYPERNYM", patterns)
				doc = nlp(TEXTS)
				matches = matcher(doc)
				for match_id, start, end in matches:

					string_id = nlp.vocab.strings[match_id]  # Get string representation
					extended_start = int(start)-25
					extended_end = int(end)+25
					span = doc[start:end]
					extended_span = doc[extended_start:extended_end]  # The matched span
					span_text = extended_span.text
					match_text = span.text
					results = string_id, start, end, span.text, extended_start, extended_end, extended_span.text
					span_index_token_start = span_text.find(match_text)
					span_index_token_end = span_index_token_start + len(match_text.split(' ')[0])
					#print(results)
					print('{"text": "'+span_text+'"}')
					resultfile = "./data.txt"
					#print(nlp.pipe_names)
					with open(resultfile, "a") as new_text_file:
						string_results = str(results)
						new_text_file.write(span_text+'\n')
			doc = None
			nlp = None

		else:
			print("error while reading directory in line 16 of script")

	except RuntimeError:
		print (str(file))
"""

			for match_id, start, end in matches:
				string_id = nlp.vocab.strings[match_id]  # Get string representation
				extended_start = int(start)-25
				extended_end = int(end)+25
				span = doc[start:end]
				extended_span = doc[extended_start:extended_end]  # The matched span
				results = match_id, string_id, start, end, span.text, extended_start, extended_end, extended_span.text
				print(results)
				resultfile = "./real_hypernym_output.txt"
				print(nlp.pipe_names)
				with open(resultfile, "a") as new_text_file:
					string_results = str(results)
					new_text_file.write(string_results)
		doc = None
		nlp = None

	else:
		print("error while reading directory in line 16 of script")

except RuntimeError:
	print (str(file))


'"{text": ' extended_span.text, ', '"spans": [{"text":'span.text[0:int(end)-int(start), ', "start":'str(span.text[0:int(end)-int(start))[0]}

{"text": "Furthermore, Smad-phosphorylation was followed by upregulation of Id1 mRNA and Id1 protein, whereas Id2 and Id3 expression was not affected.", 
"spans": [{"text": "Smad", "start": 13, "token_start": 2, "token_end": 2, "end": 17, "type": "span", "label": "GGP"}

{"text": "the text", "spans": [{"text": "Token text", "start": Int index of tokens first letter in string of text, "token_start": the int index of token in all tokens, 
"token_end": same as before - same as start if one token, if noun phrase then end on last token, "end": index of tokens last letter, "type": "span", label "HYPERNYM"}
"""

#[(token.text,token.idx) for token in parsed_sentence]

# Create a Doc object for each text in TEXTS

	# Match on the doc and create a list of matched spans
	#spans = [doc[start:end]
	# Get (start character, end character, label) tuples of matches
	#entities = [(span.start_char, span.end_char, "HYPERNYM") for span in spans]
	# Format the matches as a (doc.text, entities) tuple
	#training_example = (doc.text, {"entities": entities})
	# Append the example to the training data
	#TRAINING_DATA.append(training_example)

#print(*TRAINING_DATA, sep="\n")

# grade this against the hearst patterns as 1/2 patterns for just hyponyms with inference of downstream hyponyms
# to find in parse tree to suite SpaCy
