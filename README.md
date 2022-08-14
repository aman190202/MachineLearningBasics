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

<img src="images/naive.jpeg" width="400">

### Gaussian Naive Bayes Classifier
In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution.

A practical implementation of Gaussian Naive Bayes on the Iris Dataset can be seen [here](gaussian_naive_bayes_iris.py).

To execute it :
```
$ python gaussian_naive_bayes_iris.py
```
## Support Vector Machine

Support vector machines (SVMs, also known as support vector networks) are one of the popular algorithm in the world of machine learning. SVM is a supervised alogorithm and can be used to regression and classification problems.
The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.Two lines are drawn passing through support vecotors and this two lines is called margin line.

There can be multiple hyperplane possible but we always choose that hyperplan who's margin distance is maximum because it will help us to generalise our model better.
<img src="images/support-vector-machine-algorithm.png" width="400"><br>
Depending upon the data SVM can be of two types :

1. Linear Separable SVM

<img src="images/support-vector-machine-algorithm4.png" width="400"><br>

2. Non-Linear Seperable SVM

<img src="images/support-vector-machine-algorithm6.png" width="400"><br>

We solve Non-Linear Separable SVM by using non Linear SVC kernels. The basic task of SVC kernel is to increase the dimensionality of the model to create hyperplane.
Depending upon the feautures present in the data there can be N-Dimensional hyperplane where N stands for number of features present in the data.




## Decision Trees
Decision Trees are a type of Supervised Machine Learning where the data is continuously split according to a certain parameter. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
Decision trees consists of two nodes: the Decision Node and the Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches. The decisions or the test are performed on the basis of features of the given dataset.

<img src="images/Screen Shot 2022-08-14 at 06.30.46.png" width="400"><br>
<sub><sup>StatQuest. “Decision and Classification Trees, Clearly Explained!!!” YouTube, Joshua Starmer, 26 Apr. 2021, https://youtu.be/_L39rN6gz7Y</sup></sub>

Decision trees can be subdivided into two types:
- <b>Classification Trees</b>: Maps the binary decision that lead to a diecison about the class of an object. It labels, records and assigns variables to descrete classes. Classification data is generally preferred for decision trees.

- <b>Regression Trees</b>: It is designed to approximate continous and real-valued functions. Makes use of binary recursive partioning, which is an  iterative process that splits the data into partitions or branches, and then continues splitting each partition into smaller groups as the method moves up each branch.

To predict the class of the given dataset, the algorithm starts from the root node of the tree. This algorithm compares the values of root attribute with the record (real dataset) attribute and, based on the comparison, follows the branch and jumps to the next node. For the next node, the algorithm again compares the attribute value with the other sub-nodes and move further. It continues the process until it reaches the leaf node of the tree.

Implementation of Decision trees on the Pima Indians Diabetes Dataset can be seen [here](decision-trees.ipynb).

## Random Forest

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance.
Random forest algorithms have three main hyperparameters, which need to be set before training. These include node size, the number of trees, and the number of features sampled. From there, the random forest classifier can be used to solve for regression or classification problems.

The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample. Of that training sample, one-third of it is set aside as test data, known as the out-of-bag (oob) sample, which we’ll come back to later. Another instance of randomness is then injected through feature bagging, adding more diversity to the dataset and reducing the correlation among decision trees. Depending on the type of problem, the determination of the prediction will vary. For a regression task, the individual decision trees will be averaged, and for a classification task, a majority vote—i.e. the most frequent categorical variable—will yield the predicted class. Finally, the oob sample is then used for cross-validation, finalizing that prediction.

<img src="images/Random Forest Diagram.jpeg" width="400"><br>
<sub><sup>IBM</sup></sub>

Random Forests have the following benefits over the Decision Trees:
- <b>Reduced risk of overfitting</b>: Decision trees run the risk of overfitting as they tend to tightly fit all the samples within training data. However, when there’s a robust number of decision trees in a random forest, the classifier won’t overfit the model since the averaging of uncorrelated trees lowers the overall variance and prediction error.

- <b>Provides flexibility</b>: Since random forest can handle both regression and classification tasks with a high degree of accuracy, it is a popular method among data scientists. Feature bagging also makes the random forest classifier an effective tool for estimating missing values as it maintains accuracy when a portion of the data is missing.

- <b>Easy to determine feature importance</b>: Random forest makes it easy to evaluate variable importance, or contribution, to the model. There are a few ways to evaluate feature importance. Gini importance and mean decrease in impurity (MDI) are usually used to measure how much the model’s accuracy decreases when a given variable is excluded. However, permutation importance, also known as mean decrease accuracy (MDA), is another importance measure. MDA identifies the average decrease in accuracy by randomly permutating the feature values in oob samples.

Implementation of Random Forest on the NOAA's Climate Data for temperatures in Seattle, Washington can be seen [here](random-forest.ipynb).



## Required Libraries
```
$ pip install datascience
```
