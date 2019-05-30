# Ising Network Model in Python

The Ising network model is a type of undirected graphical model that belongs to the Markov Random Field. The Ising network model is closely related to psychometrics network models for modeling psychological traits. It is also widely used in diverse areas to model binary features from a graphical approach. However, despite its versatile application, there is no developed package in Python for the Ising network model. In this project, I developed a Python class for learning the Ising network model. This unsupervised learning algorithm estimates an Ising network model from data with binary features.

## Ising Model

## How to Use the Module
### Pre-requisite
To be able to use this module, the following packages are required:
1. numpy (http://www.numpy.org/)
2. scikit-learn version is 0.21.2. (http://scikit-learn.org/stable/install.html)
3. math (https://docs.python.org/3/library/math.html#module-math)
4. networkx (https://networkx.github.io/documentation/stable/install.html)

### Parameters

### Methods
1. __fit(self, X)__: fit the model according to the given training data
2. __score(self, X)__: return the approximate log-likelihood of the given data
3. __potential(self, X)__: compute potential of the given instances according to the learned model
4. __predict(self, X, i)__: predict the most likely state of node i given information of all other nodes

### Input Data & Output

## Examples
