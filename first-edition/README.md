Data Science from Scratch
=========================

Here's all the code and examples from the first edition of my book __[Data Science from Scratch](http://joelgrus.com/2015/04/26/data-science-from-scratch-first-principles-with-python/)__. The `code` directory contains Python 2.7 versions, and the `code-python3` direction contains the Python 3 equivalents. (I tested them in 3.5, but they should work in any 3.x.)


Each can be imported as a module, for example (after you cd into the /code directory):

```python
from linear_algebra import distance, vector_mean
v = [1, 2, 3]
w = [4, 5, 6]
print distance(v, w)
print vector_mean([v, w])
```

Or can be run from the command line to get a demo of what it does (and to execute the examples from the book):

```bat
python recommender_systems.py
```

Additionally, I've collected all the [links](https://github.com/joelgrus/data-science-from-scratch/blob/master/links.md) from the book.

And, by popular demand, I made an index of functions defined in the book, by chapter and page number.
The data is in a [spreadsheet](https://docs.google.com/spreadsheets/d/1mjGp94ehfxWOEaAFJsPiHqIeOioPH1vN1PdOE6v1az8/edit?usp=sharing), or I also made a toy (experimental) [searchable webapp](http://joelgrus.com/experiments/function-index/).

## Table of Contents

1. Introduction
2. A Crash Course in Python
3. [Visualizing Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/visualizing_data.py)
4. [Linear Algebra](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/linear_algebra.py)
5. [Statistics](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/statistics.py)
6. [Probability](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/probability.py)
7. [Hypothesis and Inference](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/hypothesis_and_inference.py)
8. [Gradient Descent](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/gradient_descent.py)
9. [Getting Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/getting_data.py)
10. [Working With Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/working_with_data.py)
11. [Machine Learning](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/machine_learning.py)
12. [k-Nearest Neighbors](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/nearest_neighbors.py)
13. [Naive Bayes](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/naive_bayes.py)
14. [Simple Linear Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/simple_linear_regression.py)
15. [Multiple Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/multiple_regression.py)
16. [Logistic Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/logistic_regression.py)
17. [Decision Trees](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/decision_trees.py)
18. [Neural Networks](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/neural_networks.py)
19. [Clustering](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py)
20. [Natural Language Processing](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py)
21. [Network Analysis](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/network_analysis.py)
22. [Recommender Systems](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/recommender_systems.py)
23. [Databases and SQL](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/databases.py)
24. [MapReduce](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/mapreduce.py)
25. Go Forth And Do Data Science
