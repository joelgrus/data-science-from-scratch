Data Science from Scratch
=========================

Here's all the code and examples from the second edition of my book _Data Science from Scratch_. They require at least Python 3.6.

(If you're looking for the code and examples from the first edition, that's in the `first-edition` folder.)

If you want to use the code, you should be able to clone the repo and just do things like

```
In [1]: from scratch.linear_algebra import dot

In [2]: dot([1, 2, 3], [4, 5, 6])
Out[2]: 32
```

and so on and so forth.

Two notes:

1. In order to use the library like this, you need to be in the root directory (that is, the directory that contains the `scratch` folder). If you are in the `scratch` directory itself, the imports won't work.

2. It's possible that it will just work. It's also possible that you may need to add the root directory to your `PYTHONPATH`, if you are on Linux or OSX this is as simple as 

```
export PYTHONPATH=/path/to/where/you/cloned/this/repo
```

(substituting in the real path, of course).

If you are on Windows, it's [potentially more complicated](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).

## Table of Contents

1. Introduction
2. A Crash Course in Python
3. [Visualizing Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/visualization.py)
4. [Linear Algebra](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/linear_algebra.py)
5. [Statistics](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/statistics.py)
6. [Probability](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/probability.py)
7. [Hypothesis and Inference](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/inference.py)
8. [Gradient Descent](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/gradient_descent.py)
9. [Getting Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/getting_data.py)
10. [Working With Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/working_with_data.py)
11. [Machine Learning](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/machine_learning.py)
12. [k-Nearest Neighbors](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/k_nearest_neighbors.py)
13. [Naive Bayes](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/naive_bayes.py)
14. [Simple Linear Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/simple_linear_regression.py)
15. [Multiple Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/multiple_regression.py)
16. [Logistic Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/logistic_regression.py)
17. [Decision Trees](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/decision_trees.py)
18. [Neural Networks](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/neural_networks.py)
19. [Deep Learning]
20. [Clustering](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/clustering.py)
21. [Natural Language Processing](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/nlp.py)
22. [Network Analysis](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/network_analysis.py)
23. [Recommender Systems](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/recommender_systems.py)
24. [Databases and SQL](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/databases.py)
25. [MapReduce](https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/mapreduce.py)
26. Data Ethics
27. Go Forth And Do Data Science
