#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:37:59 2019

@author: Avinash.OK
"""
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import warnings, pickle, spacy, sys
import xgboost as xgb
import nltk.stem.snowball
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
plt.style.use('bmh')
# Turn interactive plotting off
plt.ioff()
warnings.filterwarnings('ignore')
st = nltk.stem.snowball.SnowballStemmer('english')
stemmer = SnowballStemmer('english')
nlp = spacy.load('en')




def textClassificationFn(df, requiredColumnList, modelNumber, outputDirectory):
    """
    1. Logistic Regression: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    2. Naive Bayes: Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    3. Stochastic Gradient Descent: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    4. Random Forest Classifier : Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    5. XGBoost Classifier: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    6. Support Vector Machines(SVC) : Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
    """

    commentTextColumn = requiredColumnList['commentTextColumn']
    agentAssignedColumn = requiredColumnList['agentAssignedColumn']

    X = df[commentTextColumn]
    y = df[agentAssignedColumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    uniqueTopics = list(df[agentAssignedColumn].unique())

    if modelNumber =='1':
        algorithmName = "Logistic regression"
        print("Results of : "+algorithmName)

        logreg = Pipeline(
            [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)), ])
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))


        # save the classifier
        modelFileName = 'logisticRegressionModel.pkl'
        with open(outputDirectory+modelFileName, 'wb') as fid:
            pickle.dump(logreg, fid)
        print("The trained model of"+algorithmName+" is saved in "+outputDirectory+"as "+modelFileName)

    elif modelNumber =='2':

        algorithmName = "Naive Bayes"
        print("Results of : " + algorithmName)

        nb = Pipeline(
            [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha=0.01)), ])
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))

        # save the classifier
        modelFileName = 'naiveNayesModel.pkl'
        with open(outputDirectory + modelFileName, 'wb') as fid:
            pickle.dump(nb, fid)
        print("The trained model of" + algorithmName + " is saved in " + outputDirectory + "as " + modelFileName)


    elif modelNumber == '3':

        algorithmName = "Stochastic Gradient Descent"
        print("Results of : " + algorithmName)

        sgd = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), (
        'clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2, random_state=42, max_iter=5, tol=None)), ])
        sgd.fit(X_train, y_train)

        y_pred = sgd.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))

        # save the classifier
        modelFileName = 'stochasticGradientDescent.pkl'
        with open(outputDirectory + modelFileName, 'wb') as fid:
            pickle.dump(sgd, fid)
        print("The trained model of" + algorithmName + " is saved in " + outputDirectory + "as " + modelFileName)

    elif modelNumber == '4':

        algorithmName = "Random Forest Classifier"
        print("Results of : " + algorithmName)

        forest = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), (
        'clf', RandomForestClassifier(max_features='sqrt', n_estimators=1000, max_depth=3, random_state=0)), ])
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))

        # save the classifier
        modelFileName = 'randomForest.pkl'
        with open(outputDirectory + modelFileName, 'wb') as fid:
            pickle.dump(forest, fid)
        print("The trained model of" + algorithmName + " is saved in " + outputDirectory + "as " + modelFileName)

    elif modelNumber == '5':

        algorithmName = "XGBoost Classifier"
        print("Results of : " + algorithmName)

        xgboost = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf', xgb.XGBClassifier(n_jobs=1, max_depth=3, learning_rate=0.01, n_estimators=1000)), ])
        xgboost.fit(X_train, y_train)
        y_pred = xgboost.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))

        # save the classifier
        modelFileName = 'xgboostClassifier.pkl'
        with open(outputDirectory + modelFileName, 'wb') as fid:
            pickle.dump(xgboost, fid)
        print("The trained model of" + algorithmName + " is saved in " + outputDirectory + "as " + modelFileName)


    elif modelNumber == '6':

        algorithmName = "Support Vector Machines(SVM)"
        print("Results of : " + algorithmName)

        svc = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                        ('clf', SVC(gamma='scale', decision_function_shape='ovo'))])
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred, target_names=uniqueTopics))

        # save the classifier
        modelFileName = 'supportVectorMachine.pkl'
        with open(outputDirectory + modelFileName, 'wb') as fid:
            pickle.dump(svc, fid)
        print("The trained model of" + algorithmName + " is saved in " + outputDirectory + "as " + modelFileName)

    else:
        print("Program Ends.")
        sys.exit()

