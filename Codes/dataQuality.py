#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:37:59 2019

@author: Avinash.OK
"""
from __future__ import unicode_literals
import matplotlib.pyplot as plt
import warnings
import numpy as np
import sys, time, logging
import nltk.stem.snowball
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
plt.style.use('bmh')
# Turn interactive plotting off
plt.ioff()
st = nltk.stem.snowball.SnowballStemmer('english')
warnings.filterwarnings('ignore')
stemmer = SnowballStemmer('english')


def dataQualityFn(df, dataTypeDictionary, requiredColumnList,  outputDirectory):
    directory = outputDirectory + 'DataQuality/'
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    outputDirectory = directory

    logging.basicConfig(filename=outputDirectory + 'appDataQuality.log', filemode='w', format='PROCESS INFO-%(message)s',
                        level=logging.CRITICAL)

    # Function to print & log process information
    def printAndLogInfo(customMessage, exceptionMessage=None):
        print(customMessage)
        logging.critical(customMessage)
        if exceptionMessage:
            print(str(exceptionMessage))
            logging.critical(exceptionMessage)

    # Function to print & log errors
    def printAndLogError(customMessage, exceptionMessage=None):
        print('ERROR!!! ' + customMessage)
        logging.critical(customMessage)
        if exceptionMessage:
            print(str(exceptionMessage))
            logging.critical(exceptionMessage)
        time.sleep(10)
        sys.exit()


    agentAssignedColumn = requiredColumnList['agentAssignedColumn']
    commentTextColumn = requiredColumnList['commentTextColumn']

    duplicatesCount = {}
    for col in df.columns.to_list():
        duplicatesCount[col] = [((df.duplicated(col).sum() / len(df)) * 100),
                                100 - ((df.duplicated(col).sum() / len(df)) * 100)]

    nullCounter = {}
    for col in df.columns.to_list():
        count = 0
        for cell in df[str(col)]:
            if cell == '?' or cell == "":  # or len(str(cell))==1
                count = count + 1
        nullCounter[col] = [float(count / len(df)) * 100, 100 - float(count / len(df)) * 100]


    def dataQualityCheck(checkName, columnName):
        if checkName == "Null Values":
            # create data
            names = 'Null Values', 'Non Null Values',
            size = np.array(nullCounter[columnName])
            printAndLogInfo("Null Values Data Quality Check for " + str(columnName))

            def absolute_value(val):
                a = np.round(val / 100. * size.sum(), 0)
                return a

            # Create a circle for the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='white')

            # Custom colors --> colors will cycle
            plt.pie(size, labels=names, colors=['red', 'green'], autopct=absolute_value)
            plt.title(checkName + " Pie of " + columnName, fontdict=None, loc='center')
            p = plt.gcf()
            p.gca().add_artist(my_circle)
            plt.savefig(outputDirectory + checkName + columnName + '.png', bbox_inches='tight')
            plt.close()
        elif checkName == "Duplicates":
            # create data
            names = 'Duplicate Values', 'Unique Values Values',
            size = np.array(duplicatesCount[columnName])
            printAndLogInfo("Duplicate Value Data Quality check for " + str(columnName))

            def absolute_value(val):
                a = np.round(val / 100. * size.sum(), 0)
                return a

            # Create a circle for the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='white')

            # Custom colors --> colors will cycle
            plt.pie(size, labels=names, colors=['red', 'green'], autopct=absolute_value)
            plt.title(checkName+ " Pie of "+columnName, fontdict=None, loc='center')
            p = plt.gcf()
            p.gca().add_artist(my_circle)
            plt.savefig(outputDirectory + checkName + columnName + '.png', bbox_inches='tight')
            plt.close()
        elif checkName == "Details":
            printAndLogInfo("Details of the Column: \n ")
            printAndLogInfo("Original Datatype should be " + dataTypeDictionary[columnName] + "\n")
            printAndLogInfo("Datatype in the data is " + str(df[str(columnName)].dtypes) + "\n")
        elif checkName == "Range":
            if str(df[str(columnName)].dtypes) == 'int64' or str(df[str(columnName)].dtypes) == 'datetime64[ns]':
                printAndLogInfo("Maximum Value is " + str(df[str(columnName)].max()) + " \n ")
                printAndLogInfo("Minimum Value is " + str(df[str(columnName)].min()))
            else:
                printAndLogInfo("Since the Datatype of column " + str(
                    columnName) + " is not numeric in the given data, Range cannot be calculated.")


    def dQexecute(columnName):
        printAndLogInfo("\n Name of the Column " + str(columnName) + "\n \n")
        dataQualityCheck("Details", columnName)
        dataQualityCheck("Null Values", columnName)
        dataQualityCheck("Duplicates", columnName)
        dataQualityCheck("Range", columnName)
        printAndLogInfo("*****************")


    for col in df.columns.to_list():
        dQexecute(col)

    # Agent Assigned Topic Distribution Analysis

    uniqueTopics = list(df[agentAssignedColumn].unique())
    fig = plt.figure(figsize=(12, 5))
    df.groupby(agentAssignedColumn)[commentTextColumn].count().plot.bar(ylim=0)
    plt.savefig(outputDirectory+'labelDistribution.png', bbox_inches='tight')
    plt.close()

    df['totalwords'] = df[commentTextColumn].str.split().str.len()


    def reasonCodeLevelWordCount(reasonCode, parameter):
        dfReasonCodeSubset = df[df[agentAssignedColumn] == reasonCode]
        if parameter == 'mean':
            return float(dfReasonCodeSubset.describe()['totalwords'][1])
        elif parameter == 'median':
            return float(dfReasonCodeSubset.describe()['totalwords'][5])


    # Mean Word Count
    reasonCodeDict = {}
    for topic in uniqueTopics:
        reasonCodeDict[str(topic)] = float(reasonCodeLevelWordCount(topic, 'mean'))
    plt.figure(figsize=(20, 20))
    plt.title("Mean Word Frequency for each Topic", fontdict=None, loc='center')
    plt.bar(reasonCodeDict.keys(), reasonCodeDict.values(), width=0.1, color='g')
    plt.savefig(outputDirectory+'meanBarGraph.png', bbox_inches='tight')
    plt.close()

    printAndLogInfo("\n\n ******************** \n\n ")

    # Median Word Count (Optional)
    reasonCodeDict = {}
    for topic in uniqueTopics:
        reasonCodeDict[str(topic) ] =float(reasonCodeLevelWordCount(topic, 'median'))
    plt.figure(figsize=(20 ,20))
    plt.title("Median Word Frequency for each Topic", fontdict=None, loc='center')
    plt.bar(reasonCodeDict.keys(), reasonCodeDict.values(), width = 0.1  , color='g')
    plt.savefig(outputDirectory+ 'medianBarGraph.png', bbox_inches='tight')
    plt.close()

    # Visualize Token (vocabulary) Frequency Distribution Before Text Preprocessing
    vectorizer = CountVectorizer()
    docs = vectorizer.fit_transform(df[commentTextColumn])
    features = vectorizer.get_feature_names()
    plt.figure(figsize=(12, 8))
    plt.title("FrequencyDistribution of words before Preprocessing", fontdict=None, loc='center')
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    for label in visualizer.ax.texts:
        label.set_size(20)
    # visualizer.poof()
    plt.savefig(outputDirectory + 'FrequencyDistributionBeforePreprocessing.png', bbox_inches='tight')
    plt.close()

    #Preparation of Data Quality Report

    from fpdf import FPDF
    pdf = FPDF()
    # imagelist is the list with all image filenames
    filelist = os.listdir(outputDirectory)
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not (fichier.endswith(".png")):
            filelist.remove(fichier)
    imagelist = filelist

    for image in imagelist:
        pdf.add_page()
        pdf.image(outputDirectory+image, 40, 20, 100, 80)
    pdf.output(outputDirectory+"DataQualityReport.pdf", "F")

    printAndLogInfo("Detailed Report on Data Quality is saved in the location: "+outputDirectory)




    return df


