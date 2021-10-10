# This is a repo for the DLAI squad

## In this repository you will find the scripts we are currently using to build out the corpus, create the translator, and generate results for the manuscript.

### Each folder has it's own README and the titles of the folders should be self-explanatory.

Please note that the datasets i.e. UMLS are not included here and some of them require a license. Contact me if you need access to any of these.

Additionally, since prodigy is mostly being run on the command line, there isn't much here for that. The script that calls the models we are building is in /translate/automated_translator_v3.ipynb  
To see the exact environment I built to run these scrips see the file 'DLAI_environment.yaml'  
That file is quite robust and probs would be difficult to install on it's own so the most important packages are:  
- rapidsai  
-- https://rapids.ai/start.html#get-rapids
- CWI  
-- https://github.com/siangooding/cwi
- dask  
-- https://docs.dask.org/en/latest/install.html 
- spaCy  
-- https://spacy.io/usage
- scispaCy  
-- https://allenai.github.io/scispacy/

This requires conda.  
We use the most current versions for each as of Oct 2021, I would first install rapidsai via the generated command on their website.  
Then install CWI, dask, spaCy and scispaCy  
Any dependencies after installing these can be safely installed, first try with conda, if not available then pip.