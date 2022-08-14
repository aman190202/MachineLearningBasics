# Machine Learning Basics
To Practically implement and study the effect of different optimizers on hyperparamters
## Naive Bayes Classification
The fundamental Naive Bayes assumption is that each feature makes an:
1. independent
2. equal
contribution to the outcome.

It is importatnt to undertand [Bayes' Theorem](https://www.investopedia.com/terms/b/bayes-theorem.asp) to understand Naive Bayes.

### Bayes' Theorem
Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

<img src="images/1*LaTxXlJ0tz0dUvBqISvylw@2x.jpeg" width="400">

### Gaussian Naive Bayes Classifier
In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution.

A practical implementation of Gaussian Naive Bayes on the Iris Dataset can be seen [here](gaussian_naive_bayes_iris.py)

To execute it :
```
$ python gaussian_naive_bayes_iris.py
```
## Support Vector Machine

Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane.

An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification, implicitly mapping their inputs into high-dimensional feature spaces.

## Decision Trees


## Required Libraries
```
$ pip install datascience
```
