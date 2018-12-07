# SF_diversity
Spatial frequency diversity project

Here is all the code written in Python for the SF mixture project. The model is largely based on the model in Goris, Simoncelli, Movshon (2015). (*Origin and Function of Tuning Diversity in Macaque Visual Cortex*; **Neuron**)

The code is organized in the following way: in each sub-directory (e.g. .../LGN/), we have structures, figures, and code associated with a the particular experiment corresponding to that directory. In this main directory (SF_diversity/), we have code and wrapper functions (wrapping experiment-specific code) for running analyses (as of 12.07.18, working on making this general branch more powerful and running more and more from here rather than from the specific sub-directories.) 

Within each directory:
* The initial files within each sub-directory are the primary code for processing and analyzing data, as well as fitting models with Scipy
* code for fitting models in Tensorflow is in /tf-func/
* saved figures are in /Figures/
* data structures, lists of optimization results from various fits (be it full model, descriptive tuning fits, etc) are in /Structures/
