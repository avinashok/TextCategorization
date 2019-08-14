Text Categorization using Machine Learning
==========


Text Categorization(sometimes referred to as Text Classification) is one of the basic & most popular task in [Natural Language Processing(NLP)](https://en.wikipedia.org/wiki/Natural_language_processing). There are different approaches available for performing this both Supervised and Unsupervised.

In this project I'm trying to list down various popular approaches to do text categorization and opening a platform to compare the results of different approaches. This repository would give you a quick & overall understanding of how to approach a similar 'Text Categorization-Sentiment Analysis' type problem.

Python 3.7.3(latest version) is used to develop the code.

For development purposes, I have created a sample [dataset](https://github.com/avinashok/TextCategorization/blob/master/Data/CustomerInteractionData.csv) with text content being the agent-customer initeraction comments. I suggest you to go through ['Notebooks/TextCategorizationApproaches.ipynb'](https://github.com/avinashok/TextCategorization/blob/master/Notebooks/TextCategorizationApproaches.ipynb) for a deeper understanding about the Exploratory Data Analysis(EDA) performed on the data. You can also use [nbviewer](https://nbviewer.jupyter.org/) for viewing the JupyterNotebook.

Alternatively, for a quick Demo, use the command:

    python '/Codes/CommentCategorizationMain.py' -W ignore::DeprecationWarning

This code will enable you to iteratively compare between different approaches for the same dataset & evaluate the training accuracy of each one of them.

One of the peculiarities of this project is, I have tried to streamline the Text Preprocessing steps one by one in a best possible approach with prime focus on retaining the meaning of each text datapoints.

Heuristic/Rule-based approach which nowadays very few people use is also included in the code. Eventhough this is a primitive approach & has nothing to do with Machine Learning, at times I feel this approach can yield better results based on the type of text data you're dealing with.

The Supervised Learning approaches used in this project are:
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
- [XGBoost](https://en.wikipedia.org/wiki/XGBoost)
- [Support Vector Machine(SVC)](https://en.wikipedia.org/wiki/Support-vector_machine)
- [Combination of Word2Vec & Logistic Regression](https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)
- [Combination of Doc2Vec & Logistic Regression](https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)
- [Bag-Of-Words with Keras Sequential Model](https://keras.io/getting-started/sequential-model-guide/)
- [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)
- Heuristic/Rule Based Approach.

The Unsupervised Learning approaches used in this project are:
[Latent Dirichlet allocation(LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) using 
- Bag of Words
- TF-IDF

In trying to incorporate the context of sentences at times, we also try two word gram models:
- Unigram Word distribution
- Bigram Word distribution


Scope of Improvements:
1) Grid Search could be used to optimize various algorithm parameters, there by fitting a model with maximum accuracy.
2) [Latent Semantic Analysis(LSA)](https://en.wikipedia.org/wiki/Latent_semantic_analysis), [Restricted Boltzman Machine(RBM)](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine), [lda2vec](https://arxiv.org/abs/1605.02019) etc. can also be tried out under Unsupervised Learning approach, but would require better processing capacity to run.
3) Based on the data you're dealing with, Text Preprocessing methods & approaches may vary.
