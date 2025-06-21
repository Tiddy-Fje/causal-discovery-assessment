# Causal discovery assessment   

## Project overview  

In many rich datasets, attributing causal relations to variables in a principled way represents a challenge. Peter-Clark (PC) and Incremental Association Markov Blanket (IAMB) algorithms are commonly used algorithms in such contexts.

In this project, we compare these two methods of doing causal discovery on the LUCAS0 artificially generated lung cancer data set. We do so in the context of then running a logistic regression for the lung cancer target variable T, in terms of other variables. As baselines, we also include ground truth and LASSO variable selections. To study the stability of the methods we also vary the size of the training set.

See report.pdf for a detailed analysis and discussion of the project. It includes a quick exploration of the dataset, descriptions of the considered algorithms, and the results associated to logistic regression performance as well as sensitivity to the size of the training set.

## Running the code 

To manage the `python` packages needed to run the files `conda` was used. The `requirements.yml` file can be used to create the associated environment easily as `conda create --n <env-name> --file <relative-path-to-this-file>` (or using similar non-`conda` commands).

`plt_params.py` only fixes some common font sizes for the figure generation. It is imported in the other files.
`causal_discovery.py` and `lasso.py` are used to run the selection of the Markov blankets with the different methods and generate the corresponding figures. `logistic.py` trains the logistic regressions for different estimates of the Markov blanket and then compares their performance by computing AUC scores, BIC, and p-values. It also generates the table included in the report. All these files can be run as `python <filename>.py`. 

## Generating the report PDF
To generate a pdf of the report, standard `LaTeX` command line prompts of your local machine apply. 

## Acknowledgements
This project was developed as part of the Applied Statistics course at EPFL. I thank Dr. Linda Mhalla for providing the project statement and initial guidance. I also thank Rayan Harfouche for valuable discussions. 

## Note about Git history
This project was initially submitted to a Github classroom repo. This version is a copy of that submission that is posted on my personal Github account. I could unfortunately not recover the original Git history.  