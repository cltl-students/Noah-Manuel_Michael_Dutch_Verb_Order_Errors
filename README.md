# ma_thesis
### Thesis repository. Noah-Manuel Michael. VU Amsterdam. 2023.

The repository contains the following directories and files:
- Data
- Parsers
- Transformers
- requirements.txt

---------------

The /Data directory contains the following directories:
- Annotation
- Dataset_Construction

The /Annotation directory contains the annotation from both first-language speakers as well as the code used for annotating.
It contains the extracted learner sentences from the Leuven corpus (cannot be made publicly available), information about
metadata and scripts for preprocessing of the Leuven corpus.

The /Dataset_Construction directory contains the untreated seed corpora for the synthetic datasets, scripts for
preprocessing the data, scripts for permuting the preprocessed data and the resulting datasets (only the test data,
the train data files are too large to be uploaded on GitHub but are available upon request).

---------------

The /Parsers directory contains the following directories:
- Detection

The /Detection directory contains the following:
- /Classifiers: the classifiers and vectorizers trained based on the output of the PoS tagger and the parser
- /Data: 
    - /Trees: The trees of the Pool and test sets
    - /Tuples: The tuples that serve as input for the PoS and parser classifier approaches, and the predictions
- A variety of scripts for parsing and processing both the train and test data for the Parse Lookup, PoS Classifier, and Parse Classifier approaches

Results for the Lookup approach can be obtained by running the scripts:
- ex1_pool_of_parses.py
- ex2_pool_of_simplified_parses.py

Results for the PoS and Parse Classifier approaches can be obtained by running the script:
- results_parsers.py

---------------

The /Transformers directory contains the following directories:

- Detection

The /Detection directory contains the following:
- /BERTJe: code for fine-tuning BERTje and prediction
- /GPT-2: code for fine-tuning GPT-2 and prediction
- /RobBERT: code for fine-tuning RobBERT and prediction
- /Predictions: the predictions on Rand, Verbs, Info (I)
- /Predictions2: the predictions on Rand, Verbs, Info (II)
- /Predictions3: the predictions on Rand, Verbs, Info (III)
- /Predictions_Learn: the predictions on Learn (I, II, III)
- A variety of scripts for computing the results of the transformer experiments, and for sampling the pool of training data to limit the data the transformer models are trained on to 1,000,000 sentences

The code for the transformer models was predominantly run on the Surf Research Cloud. Code in the notebooks needs manual updating of filepaths if to be rerun.

---------------

Additional comments:
The disco-dop parser only runs in Linux OS. Refer to: https://github.com/andreasvc/disco-dop for details on installation.