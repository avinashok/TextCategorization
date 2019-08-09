#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:37:59 2019

@author: Avinash.OK
"""
from __future__ import unicode_literals
import warnings, spacy, logging, time, sys, os, operator, collections, re
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk.stem.snowball
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from spacy_langdetect import LanguageDetector
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.ioff()
warnings.filterwarnings('ignore')
st = nltk.stem.snowball.SnowballStemmer('english')
stemmer = SnowballStemmer('english')
nlp = spacy.load('en')


def textPreprocessingFn(df, requiredColumnList, outputDirectory):
    directory = outputDirectory + 'TextPreprocessing/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    outputDirectory = directory

    logging.basicConfig(filename=outputDirectory + 'appTextPreprocessing.log', filemode='w',
                        format='PROCESS INFO-%(message)s',
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

    commentTextColumn = requiredColumnList['commentTextColumn']
    primaryKeyColumn = requiredColumnList['primaryKeyColumn']
    agentAssignedColumn = requiredColumnList['agentAssignedColumn']

    abbrevationDictionary = {'Cus': 'customer', 'cus': 'customer',
                             'Xferred': 'transferred', 'xferred': 'transferred'}


    # Function to Standardize Text
    def objectStandardization(input_text):
        words = str(input_text).split()
        new_words = []
        for word in words:
            word = re.sub('[^A-Za-z0-9\s]+', ' ', word)  # remove special characters
            if word.lower() in abbrevationDictionary:
                word = abbrevationDictionary[word.lower()]
            new_words.append(word)
        new_text = " ".join(new_words)
        return new_text


    df[commentTextColumn] = df[commentTextColumn].apply(objectStandardization)


    # Function to extract Names of persons, organizations, locations, products etc. from the dataset
    def entityCollector(df):
        listOfNames = []
        for index, row in df.iterrows():
            doc = nlp(row[str(commentTextColumn)])
            fil = [(i.label_.lower(), i) for i in doc.ents if i.label_.lower() in ["person", "gpe",
                                                                                   "product"]]  # Extracts Person Names, Organization Names, Location, Product names
            if fil:
                listOfNames.append(fil)
            else:
                continue
        flat_list = [item for sublist in listOfNames for item in sublist]
        entityDict = {}
        for a, b in list(set(flat_list)):
            entityDict.setdefault(a, []).append(b)
        return entityDict


    entityDict = entityCollector(df)

    printAndLogInfo("\n Types of entities present in the data are: " + ", ".join(list(entityDict.keys())) + " \n")


    for entity in list(entityDict.keys()):
        entityDict[entity] = [str(i) for i in entityDict[entity]]

    ignoreWords = []
    for key in entityDict.keys():
        ignoreWords.append(entityDict[key])
    ignoreWords = [item for sublist in ignoreWords for item in sublist]

    printAndLogInfo("Number of words in Custom Stopword list = " + str(len(ignoreWords)))


    def languageDistribution(df):
        nlp = spacy.load("en")
        nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
        df['language'] = ''
        language = []
        for index, row in df.iterrows():
            text = row[str(commentTextColumn)]
            doc = nlp(text)
            language.append(str(doc._.language['language']))
        df['language'] = language
        return df


    df = languageDistribution(df)
    langDict = df.groupby('language')[str(primaryKeyColumn)].nunique().to_dict()

    otherLanguagesList = list(langDict.keys()).remove('en')
    printAndLogInfo("Some sample other language texts: \n")
    for lang in list(langDict.keys()):
        printAndLogInfo(str(df[df['language'] == str(lang)].values.tolist()[0]))


    # Dropping only the row with Spanish text
    df = df.drop(df[df['language'] == 'es'].index);

    # Function to extract Alpha numeric words
    def alphanumericExtractor(input_text):
        words = str(input_text).split()
        alphanumericWordlist = []
        for word in words:
            word = re.sub('[^A-Za-z0-9\s]+', '', word.lower())  # remove special characters
            word = re.sub(r'[^\x00-\x7F]+', ' ', word)  # remove ascii
            if not word.isdigit() and any(ch.isdigit() for ch in word):
                alphanumericWordlist.append(word)
            else:
                continue
        return alphanumericWordlist


    # Function to get the frequency of Alphanumeric words in the data
    def alphanumericFrequency(df, commentTextColumnName):
        alphanumericWordsList = []
        for index, row in df.iterrows():
            if alphanumericExtractor(row[str(commentTextColumnName)]):
                alphanumericWordsList.append(alphanumericExtractor(row[str(commentTextColumnName)]))
            else:
                continue
        flat_list = [item for sublist in alphanumericWordsList for item in sublist]
        counts = Counter(flat_list)
        countsdict = dict(counts)
        return countsdict


    # Final list of alphanumeric words
    alphanumericWordFreqDict = alphanumericFrequency(df, commentTextColumn)

    # To plot the distribution
    totalWordcount = len(alphanumericWordFreqDict)



    if totalWordcount<10:
        topWordCount = totalWordcount
    else:
        topWordCount = 10

    alphanumericWordFreqDictTop = dict(
        sorted(alphanumericWordFreqDict.items(), key=operator.itemgetter(1), reverse=True)[:int(topWordCount)])
    printAndLogInfo(alphanumericWordFreqDictTop)

    plt.figure(figsize=(20, 20))
    plt.title('Frequency of AlphaNumeric Words in the Dataset', fontdict=None, loc='center')
    plt.bar(alphanumericWordFreqDictTop.keys(), alphanumericWordFreqDictTop.values(), width=0.1, color='b');
    plt.savefig(outputDirectory + 'topAlphanumericWords.png', bbox_inches='tight')

    # Updating Custom stopword list with Alphanumeric words
    ignoreWords = ignoreWords + list(alphanumericWordFreqDict.keys())


    def clean_text(newDesc):
        newDesc = re.sub('[^A-Za-z\s]+', '', newDesc)  # remove special characters
        newDesc = re.sub(r'[^\x00-\x7F]+', '', newDesc)  # remove ascii
        newDesc = ' '.join([w for w in newDesc.split() if len(w) > 1])
        newDesc = newDesc.split()
        cleanDesc = [str(w) for w in newDesc if w not in ignoreWords]  # remove entity names, alphanumeric words
        return ' '.join(cleanDesc)


    df[commentTextColumn] = df[commentTextColumn].apply(clean_text)


    def textAutocorrect(df, columnName):
        df[str(columnName)] = df[str(columnName)].apply(lambda txt: ''.join(TextBlob(txt).correct()))
        return True

    textAutocorrect(df, commentTextColumn)

    stops = nlp.Defaults.stop_words
    default_stopwords = stopwords.words('english')
    customStopWords = {'PRON', 'pron'}
    stops.update(set(default_stopwords))
    stops.update(set(customStopWords))


    def normalize(comment, lowercase, remove_stopwords):
        if lowercase:
            comment = comment.lower()
        comment = nlp(comment)
        lemmatized = list()
        for word in comment:
            lemma = word.lemma_.strip()
            if lemma:
                if not remove_stopwords or (remove_stopwords and lemma not in stops):
                    lemmatized.append(lemma)
        normalizedSentence = " ".join(lemmatized)
        normalizedSentence = re.sub('[^A-Za-z\s]+', '', normalizedSentence)  # remove special characters
        normalizedSentence = normalizedSentence.split()
        cleanDesc = [str(w) for w in normalizedSentence if w not in stops]  # remove PRON
        return " ".join(cleanDesc)


    df[commentTextColumn] = df[commentTextColumn].apply(normalize, lowercase=True, remove_stopwords=True)


    # Removing Null Comments
    def removeNullValueCommentText(df, columnName):
        initialLength = len(df)
        df = df[pd.notnull(df[columnName])]
        finalLength = len(df)
        printAndLogInfo("\n Number of rows with Null Value in the column '" + str(columnName) + "' are: " + str(
            initialLength - finalLength))
        return df


    df = removeNullValueCommentText(df, commentTextColumn)


    # Removing duplicate comments keeping the first one
    def removeDuplicateComments(df, columnName, agentAssignedColumn):
        initialDf = df.copy()
        initialLength = len(initialDf)
        finalDf = df.drop_duplicates(subset=[columnName], keep='first')
        finalLength = len(finalDf)
        printAndLogInfo("\n Number of rows with duplicate comments in the column '" + str(columnName) + "' are: " + str(
            initialLength - finalLength))
        printAndLogInfo("\n The Level 3 Reason Codes for the dropped rows are given below: \n")
        droppedDF = initialDf[~initialDf.apply(tuple, 1).isin(finalDf.apply(tuple, 1))]
        printAndLogInfo(droppedDF[agentAssignedColumn].value_counts())
        return finalDf, droppedDF


    df, droppedDF = removeDuplicateComments(df, commentTextColumn, agentAssignedColumn)


    # Removing comments with just one word. (Like #CALL?)
    def removingShortComments(df, columnName, agentAssignedColumn, numberOfWords=1):
        initialDf = df.copy()
        initialLength = len(initialDf)
        finalDf = df[~(df[str(columnName)].str.split().str.len() < (int(numberOfWords) + 1))]
        finalLength = len(finalDf)
        printAndLogInfo("\n Number of rows with short comments in the column '" + str(columnName) + "' are: " + str(
            initialLength - finalLength))
        printAndLogInfo("\n The Level 3 Reason Codes for the dropped rows are given below: \n")
        droppedDF = initialDf[~initialDf.apply(tuple, 1).isin(finalDf.apply(tuple, 1))]
        printAndLogInfo(droppedDF[agentAssignedColumn].value_counts())
        return finalDf, droppedDF


    df, droppedDF = removingShortComments(df, commentTextColumn, agentAssignedColumn)


    vectorizer = CountVectorizer()
    docs = vectorizer.fit_transform(df[commentTextColumn])
    features = vectorizer.get_feature_names()
    plt.figure(figsize=(12, 8))
    plt.title("FrequencyDistribution of words after Preprocessing", fontdict=None, loc='center')
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    for label in visualizer.ax.texts:
        label.set_size(20)
    # visualizer.poof()
    plt.savefig(outputDirectory + 'FrequencyDistributionAfterPreprocessing.png', bbox_inches='tight')
    plt.close()


    def wordFrequency(reasonCode):
        return (df[df[agentAssignedColumn] == str(reasonCode)][commentTextColumn].str.split(
            expand=True).stack().value_counts())


    def wordFrequencyListPlot(reasonCode, plot=False):
        wordFreqDict = df[df[agentAssignedColumn] == str(reasonCode)][commentTextColumn].str.split(
            expand=True).stack().value_counts().to_dict()
        wordFreqDictMostCommon = dict(
            collections.Counter(wordFreqDict).most_common(10))  # Considering only Top 10 words
        printAndLogInfo(list(wordFreqDictMostCommon.keys()))
        if plot == True:
            plt.title(str(reasonCode), fontdict=None, loc='center')
            plt.bar(wordFreqDictMostCommon.keys(), wordFreqDictMostCommon.values(), width=0.1, color='b');
            plt.figure(figsize=(10, 10))
            plt.savefig(outputDirectory + 'wordFrequencyFor' + reasonCode + '.png', bbox_inches='tight')
            plt.close()
        return list(wordFreqDictMostCommon.keys())

    uniqueTopics = list(df[agentAssignedColumn].unique())

    for reasoncode in uniqueTopics:
        printAndLogInfo(reasoncode)
        wordFrequencyListPlot(reasoncode, plot=True)


    def wordCloudGenerator(df, reasonCode, save=False):
        dfReasonCodeSubset = df[df[agentAssignedColumn] == reasonCode]
        wordcloud = WordCloud(max_words=50, background_color='white', max_font_size=50, width=100, height=100).generate(
            ' '.join(dfReasonCodeSubset[commentTextColumn]))
        plt.imshow(wordcloud)
        plt.title(str(reasonCode), fontdict=None, loc='center')
        plt.figure(figsize=(50, 50))
        plt.axis("off")
        plt.close()
        if save:
            plt.savefig(outputDirectory+'wordCloud' + str(reasonCode) + '.png', bbox_inches='tight')


    for topic in uniqueTopics:
        wordCloudGenerator(df, topic)  # ,save = True , if you want to save the Word Clouds

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
        pdf.image(outputDirectory + image, 40, 20, 100, 80)
    pdf.output(outputDirectory + "TextPreprocessing.pdf", "F")

    printAndLogInfo("Detailed Report on Text Preprocessing is saved in the location: " + outputDirectory)

    return df