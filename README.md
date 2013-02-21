# Project 3: latent Dirichlet allocation (LDA)

## Goals
Use LDA models with the Gibbs sampling training algorithm to classify documents use a set list of classes (e.g. sports, politics).

### Method
* model: LDA
* training: Gibbs sampling 

### Implementation goals:
1. make inner loop of LDA fast
2. write function to print words with highest probability for each topic
3. write function to visualize documents based on the topics of the trained model (possible 3D)


## Datasets
Apply LDA to two datasets.

### classic400.mat
The 2D array 'classic400' contains the number of times each word is found in each document (rows=document, columns=word in vocalubary). Example: classic400(1,1) is the count of word 1 in document 1, classic400(100,500) is the count of word 500 in document 100, etc.

The array *truelabels* shows which of three domains each topic came from and can be used as a check to see if the LDA is picking up the correct classifications.

### Second Dataset (of our choice)
* our choice
* idea #1: interesting collection of documents
* idea #2: non-text dataset for which LDA model is appropriate


## Report
Report should try to answer this questions (which do not have definitive answers):
1. What is a sensible way to define and compute the goodness-of-fit, for a given dataset, of LDA models with different hyperparameters K, alpha, and beta?
2. How can you determine whether an LDA model is overfitting its training data?

For the two datasets, present and justify good values for K, alpha, and beta. The values can be chosen informally, but we need justify our choices.
