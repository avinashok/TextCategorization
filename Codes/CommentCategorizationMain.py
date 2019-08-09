#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:37:59 2019

@author: Avinash.OK
"""


from __future__ import unicode_literals

import dataQuality as dataQuality
import textPreprocessing as textPreprocessing
import textClassification as textClassification

import matplotlib.pyplot as plt
import warnings, time, sys, logging, spacy
import pandas as pd
import nltk.stem.snowball
from nltk.stem import SnowballStemmer
plt.style.use('bmh')
# Turn interactive plotting off
plt.ioff()
nlp = spacy.load('en')
stemmer = SnowballStemmer('english')
st = nltk.stem.snowball.SnowballStemmer('english')
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Directories
dataDirectory = "../Data/"
outputDirectory = "../Output/"
logging.basicConfig(filename=outputDirectory+'app.log', filemode='w', format='PROCESS INFO-%(message)s', level=logging.CRITICAL)


# Function to print & log process information
def printAndLogInfo(customMessage,exceptionMessage=None):
    print(customMessage)
    logging.critical(customMessage)
    if exceptionMessage:
        print(str(exceptionMessage))
        logging.critical(exceptionMessage)

# Function to print & log errors
def printAndLogError(customMessage,exceptionMessage=None):
    print('ERROR!!! ' +customMessage)
    logging.critical(customMessage)
    if exceptionMessage:
        print(str(exceptionMessage))
        logging.critical(exceptionMessage)
    time.sleep(10)
    sys.exit()

start_time = time.time()
printAndLogInfo("Start " + time.strftime('%Y-%m-%d %H:%M:%S'))
printAndLogInfo("Default location for files is  "+dataDirectory)

fileName = input('Please input the file name: (Please make sure the file is present in '+dataDirectory+' ) \n')

# reading data file
df = pd.read_csv(dataDirectory+fileName)

#Randomly shuffling data
df = df.sample(len(df))


#Objects Used

agentAssignedColumn = input('Name of column with Labels/Classes: ')
rawText = input('Name of column with Text comment: ')
commentTextColumn = 'CustomerInteractionText'

#Copying the Raw comment column for Future Use
df[commentTextColumn] = df[rawText]


# Identifying Primary Key
printAndLogInfo("Number of Columns in the Dataset = " + str(len(df.columns)))
df.columns.to_list()
uniqueColumns=[]
for col in df.columns.to_list():
    if len(df[str(col)].unique())==df.shape[0]:
        uniqueColumns.append(col)
if len(uniqueColumns)==1:
    primaryKeyColumn = str(uniqueColumns[0])
    printAndLogInfo("Primary Key = "+primaryKeyColumn)
else:
    primaryKeyColumn = input('Name of the Primary key column: ')


requiredColumnList = {'primaryKeyColumn': primaryKeyColumn, 'commentTextColumn':commentTextColumn,
                      'rawText': rawText, 'agentAssignedColumn': agentAssignedColumn}

df = df[list(requiredColumnList.values())]

#Data Quality
dataQualityBool = input('Do you want to perform Data Quality? (y/n) \n')

if dataQualityBool == 'y' or dataQualityBool == 'Y':
    # Specifying datatypes we want for each column
    dataTypeDictionary = {
        primaryKeyColumn: 'int64',
        commentTextColumn: 'object',
        rawText: 'object',
        agentAssignedColumn: 'object'

    }

    df = dataQuality.dataQualityFn(df, dataTypeDictionary, requiredColumnList,  outputDirectory)

elif dataQualityBool == 'n' or dataQualityBool == 'N':
    pass
else:
    printAndLogError("That was not a valid response.")
    time.sleep(10)
    sys.exit()

#Text Preprocessing
textPreprocessingBool = input('Do you want to perform Text Preprocessing? (y/n)')

if textPreprocessingBool =='y' or textPreprocessingBool =='Y':
    df = textPreprocessing.textPreprocessingFn(df, requiredColumnList, outputDirectory)

elif textPreprocessingBool =='n' or textPreprocessingBool =='N':
    pass
else:
    printAndLogError("That was not a valid response.")
    time.sleep(10)
    sys.exit()


modellingBool = input("Do you want to perform Modelling? (y/n). Type 'exit' to quit.")
printAndLogInfo("Total Number of Words in the column "+commentTextColumn+" is "+str(df[commentTextColumn].apply(lambda x: len(x.split(' '))).sum()))


while modellingBool!='exit':

    if modellingBool=='y' or modellingBool=='Y':
        modelNumber = input(
            " Among the options below, which modelling technique would you prefer now?: \n  1. LogisticRegression \n 2. Naive Bayes Classifier \n 3. SGD \n 4. Random Forest Classifier \n 5. XGBoost Classifier \n 6. Support Vector Machines. \n  7.  exit \n ")


        if modelNumber != 'exit':
            textClassification.textClassificationFn(df, requiredColumnList, modelNumber, outputDirectory)
        else:
            sys.exit()


    elif modellingBool=='n' or modellingBool=='N':
        pass
    else:
        printAndLogError("That was not a valid response.")
        time.sleep(10)
        sys.exit()

printAndLogInfo("Program Ends.")
elapsed_time = time.time() - start_time
printAndLogInfo("End " + time.strftime('%Y-%m-%d %H:%M:%S'))
printAndLogInfo('elapsed time ' + str(elapsed_time) + " seconds")
"""
1. Logistic Regression: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
2. Naive Bayes: Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
3. Stochastic Gradient Descent: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
4. Random Forest Classifier : Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
5. XGBoost Classifier: We use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
6. Support Vector Machines(SVC) : Here also, we use a pipeline approach of both Count Vectorizer and TF-IDF Vectorizer.
"""








