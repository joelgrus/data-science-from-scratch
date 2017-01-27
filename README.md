# Data Science from Scratch — Go ports

This repository contains examples from the book [_Data Science from Scratch_](http://joelgrus.com/2015/04/26/data-science-from-scratch-first-principles-with-python/) ported from Python 3 to Go.

## Running the original Python code

Install [miniconda for Python ≥ 3.5](http://conda.pydata.org/miniconda.html), then create and activate an environment with the `matplotlib` package installed:

    $ conda create --name DS_Scratch matplotlib
    $ source activate DS_Scratch
    (DS_Scratch) $

If the conda environment is successfully activated, the name of the environment `(DS_Scratch)` will be prefixed to the shell prompt, as shown above.

For chapter 3, _Visualizing Data_, the only required package is `matplotlib`. You can run the first example of chapter 3 like this:

    (DS_Scratch) $ cd 03_Visualizing_Data/
    (DS_Scratch) $ python 1_simple_line.py
