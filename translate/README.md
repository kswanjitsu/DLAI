# Automated Interpretation of Medical Text

List of packages/dependencies needed to run this script:
1. spaCy
2. scispaCy
3. CWI 
4. GingerIt
5. Tensorflow
6. Pandas
7. sklearn

## Usage
*This pipeline after some import statements runs as follows:*

**Step 1. f(x) find_complex_words:**
- Identify complex words in a document with the CWI tool
- In order to use the complex word models you must download the sequence labeler files available [here](https://github.com/marekrei/sequence-labeler), please cite both the sequence labeller paper and CWI sequence labelling paper if using these models for research.
- Additionally, the CWI method uses tensorflow < 2.0.0, so if you install from git source above, then you must open the labeler.py script and replace *import tensorflow* with the following:
- *import tensorflow.compat.v1 as tf*
- *tf.disable_v2_behavior()*

**Step 2. f(x) sub_umls_hypernyms:**
- Then use spaCy, scispaCy to perform basic NLP tasks and substitute out hypernyms, as follows
- Iterate over tokens identified as biomedical term entity with scispaCy NER model, since some of these are wordphrases, we turn that phrase into a list of words
- Then by iterating over the complex word list from Step 1 we look to see if that word is in the token
- If that token contains a complex word, we need to simplify it
- We simplify the word by searching UMLS for the word with TFIDF ngram character matching
- Once the word is found for its best match we pull the corresponding TUI for that word
- UMLS TUIs link semantic types, aka the bottom most nodes / root hypernyms in UMLS semantic relation tree
- The semantic tree is stored as a pandas dataframe (see the cell that creates semantic_df)
- Replace complex words in the sentence with hypernyms

**Step 3. f(x) resolve_abbreviations**
- Need to figure this out, scispaCy is not handling these as well, maybe steal methods from Nature paper
- May need to do this part first, then run through CWI and scispacy NER

**Step 4. f(x) runGinger:**
- We then use a grammar checker to fix errors in the newly translated text with the free API from GingerIt
- Since this is the free version it is limited to 300 characters maximum
- For this reason we chunk the sentence if it is >300 characters and rejoin it after it is checked for grammar.

**Step 5. f(x) grade_the_documents:**
- Grade the readability of the pre-substitute and post-substituted document
- We do this by importing our customized readability script, which allows it to be used with spaCy 3.+

**Step 6. f(x) automated_translation:**
- This is the main function of the script and it calls all the other functions
- I set it up this way to eventually be parallelized and hopefully some of that will go on the GPU
- This calls the functions above in order for any medical text aka "a document"
- You can run this on a single test document, or create a for loop to run over multiple documents in a file
- If you run this on multiple documents, it is extremely computationally expensive

**Citation**:
**This code uses and exemplifies each function from CWI in the `Complexity_labeller class`, from the CWI method first described in:**
*Complex Word Identifier from the paper: Complex Word Identification as a Sequence Labelling Task, 2019,* Authors: Gooding, Sian and Kochmar, Ekaterina

**Citation**:
**This code uses a sequence labeling methods first described in:**
*Semi-supervised multitask learning for sequence labeling, 2017,* Authors: Rei, Marek

## Notes for TF warnings:
- If you see warnings from TF this is because we are using TF >1.0.0 but <2.0.0, so it sees it as deprecated behavior
- This is why as described above we correct the CWI script to call compatibility with TF v1.0+ < 2.0
- If you edit this script you must restart the cluster or else TF will break due to word embeddings already being present

## Notes for "Your CPU supports instructions...:" warnings:
- This is because we are using a pre-compiled scispaCy model
- You can ignore these, building from source is pretty intense, I tried it and had to fresh install my OS

## Notes for CWI usage
**There are two options when converting text to CoNLL-type tab-separated format:**
- convert_format_string, convert_format_token
- Complexity_labeller.convert_format_string(model, 'You can convert a string like this')
- Complexity_labeller.convert_format_token(model, ['You','can','convert','tokens','like','this'])

**Once the text has been converted there are four methods to access complexity information:**
- `get_dataframe`, `get_bin_labels`, `get_prob_labels`

**The `get_dataframe` method returns a dataframe containing the original tokenized sentence, binary complexity labels and complex class probabilities.**
- If a word recieves a binary label = 1, it has been classified as a complex word.
- dataframe = Complexity_labeller.get_dataframe(model)
- Access binary labeling information from the dataframe format:
- cw_list = list(zip(dataframe['sentences'].values[0], dataframe['labels'].values[0], dataframe['probs'].values[0]))

    # get_bin_labels returns the binary complexity labels for the input
    # bin_label_list = Complexity_labeller.get_bin_labels(model)

    # The `get_prob_labels` method returns the probability of each token belonging to the complex class.
    # prob_label_list = Complexity_labeller.get_prob_labels(model)

**After identifying complex words with the CWI:**

**This script uses various tools from Explosion's spaCy and AllenAI's scispaCy in combination with wordnet to substitute complex words with their less complex hypernyms**