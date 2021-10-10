import pandas as pd
import numpy as np
import random

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

print('defining columns')
#mrconso_columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS']
mrconso_columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPRESS', 'CVF']
#mrrel_columns = ['CUI1', 'REL', 'CUI2', 'RELA', 'SAB', 'SL', 'MG']
mrrel_columns = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL','RG','DIR','SUPPRESS','CVF']

print('reading MRCONSO file into memory')
mrconso_iter = pd.read_csv("s3://sagemaker-studio-757570088617-lvhv3fp3v5/MRCONSO.RRF", iterator=True, sep='|', lineterminator='\n',names=mrconso_columns,index_col=False,encoding = "ISO-8859-1", chunksize=1000)
#print(mrconso_iter.head())
print('reconstructing DF MRCONSO for english only terms')
mrconso = pd.concat([chunk[chunk['LAT'] == 'ENG'] for chunk in mrconso_iter])

print('reading MRRELDF_single parquet file into memory')
mrrel_iter = pd.read_csv("s3://sagemaker-studio-757570088617-lvhv3fp3v5/MRREL.RRF", sep='|', iterator=True, lineterminator='\n',names=mrrel_columns,index_col=False,encoding = "ISO-8859-1", chunksize=1000)
print('reconstructing DF MRREL for "is a" relationships only aka hypernymy')
mrrel = pd.concat([chunk[chunk['RELA'] == 'isa'] for chunk in mrrel_iter])
print('removing duplicate relations from MRREL as some may be retained for different languages')
mrrel.drop_duplicates(subset = "CUI1", inplace = True)


"""
print('defining columns')
mrconso_columns = ['CUI', 'LAT', 'TS', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPRESS', 'CVF']
mrrel_columns = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL','RG','DIR','SUPPRESS','CVF']

print('reading MRCONSODF_single parquet file into memory')
mrconso_iter = pd.read_parquet("./MRCONSODF_single")

print('reconstructing DF MRCONSODF_single for english only terms')
mrconso = mrconso_iter[mrconso_iter['LAT'] == 'ENG']

print('reading MRRELDF_single parquet file into memory')
mrrel_iter = pd.read_parquet("./MRRELDF_single")

print('reconstructing DF MRCONSODF_single for "is a" relationships only aka hypernymy')
mrrel = mrrel_iter[mrrel_iter['RELA'] == 'isa']

print('removing duplicate relations from MRREL as some may be retained for different languages')
mrrel.drop_duplicates(subset = "CUI1", inplace = True)
"""

print('running main loop to get all hypernymy pairs - thanks Larry')
punctuation = [",", ":", ";"]
with open("./NER_output.txt", "w") as new_file:
	for row in mrrel.itertuples():
		cui1 = row[1]
		cui2 = row[5]
		pd1 = mrconso.loc[mrconso['CUI'] == cui1]
		pd2 = mrconso.loc[mrconso['CUI'] == cui2]
		w1 = ""
		w2 = ""
		for r in pd1.itertuples():
			s = r[15]
			if (s):
				w1 = s
				break
		for r in pd2.itertuples():
			s = r[15]
			if (s):
				w2 = s
				break
		for pattern in patterns:
			phrase = []
			POS_encountered = False
			for part in pattern:
				key = list(part.keys())[0]
				if (key == 'POS'):
					if (POS_encountered):
						phrase.append(' ' + w2)
					else:
						phrase.append(' ' + w1)
						POS_encountered = True
				elif (key == 'IS_PUNCT'):
					ind = random.randint(0, 2)
					phrase.append(punctuation[ind])
				else:
					phrase.append(' ' + part[key])
			phrase[0] = phrase[0][1: len(phrase[0])]
			phrase.append('\n')
			new_file.write("".join(phrase))