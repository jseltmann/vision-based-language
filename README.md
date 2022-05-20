# vision-based-language
My MSc Thesis project.
Language models are trained on text data (surprise!). However, the information in that text data does not have to correspond to the real world. Therefore, the knowledge learned by these models might be incorrect in some aspects. Vision-and-language data, by contrast, is grounded in the real world. In this project, I extracted text data from vision-and-language datasets and tested both language and vision-and-language models on it.

The models are tested on pairs of examples. In each pair contains an original text and a perturbed text, which is changed in some way; usually by replacing one word with a distractor.

TODO: add summary of results
TODO: add examples

## Datasets

## Generation of data
The scripts for generating the data are contained in the `generate_suites` folder. The main files in that folder are:
* `foil_image_selectors.py` contains different functions that select two images from each dataset. From these two images, the original and the distractor word will be selected.
* `context_generators.py` contains different functions that provide the part of the sentence that does not change between the original and perturbed sentences.
* `combinators.py` contains functions that combine that selected words and the context into full sentences.
* `generation_combinations_pairwise.csv` contains all the different combinations of selectors, generators, and combinators, and other conditions/restrictions, that are used to generate the different test sets.

To run the generation, adapt the file paths in `pairwise.config` (The `suites_dir` entry gives the directory, where the generated datasets will be saved). and run `python generate_pairwise.py`.

## Running tests
The files for the evaluation are in the `eval_binary` folder.
There are two different ways of evaluation:
* As a binary classification problem.
* With [syntaxgym](https://syntaxgym.org/), where each sentence in each pair is evaluated without showing the other sentence to the model.

### Binary classification
For this, you first need to train the binary classifiers for each model. For this, run the train scripts in the subfolder for each model, e.g. `train_bert.py` for BERT. (Adapt the file paths in the scripts before.)
Then, adapt the paths in `eval_binary.py` and run it.

### Syntaxgym
Coming soon ...
