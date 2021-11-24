#Alex King 2021
#Naivebayes function training testing functions and also tokenizers for 
#stopwords and binary
import os
import pandas as pd
import collections
import re
import numpy as np


def trainnaivebayes(trainClasses, vocab):
    #cancer class first
    classes = trainClasses
    logPrior = []
    numdoc = len(classes[0]) + len(classes[1])
    unique = []
    #makes vocab number unique
    for word in vocab:
        if word not in unique:
            unique.append(word)
    loglikehood = []
    #goes through each class to find probiitlies of the words given class
    for c in classes:
        numclass = len(c) #total docs for class
        logPrior.append(np.log(numclass/numdoc))   #logprior
        bigdoc = [] #all docs in class
        for docs in c: 
            bigdoc += docs
        class_count = collections.Counter(bigdoc) #counts each time a word occurs in class
        wordsum = sum(class_count.values()) #number or words in class
        
        logDict = {}
        loglikehood.append(logDict)
        for word in unique:
            try:
                logDict[word] = np.log((class_count[word] + 1) / (wordsum +1))
            except:
                logDict[word] = np.log((0 + 1) / (wordsum +1))

    return logPrior , loglikehood , unique

def testclass(testclass, logPrior, loglikehood, trainClasses, unique):
    #used to test the classes by going through the docs in the classes
    results = []
    for doc in testclass:
        results.append(testNaiveBayes(doc,logPrior,loglikehood,trainClasses,unique))
    return results #returns a list that labels the doc as cancer or not

def testNaiveBayes(testDoc, logPrior, loglikehood, trainClasses, unique):
    total = [0,0] #cancer is index 0 and 1 is noncancer
    i = 0
    j = 0
    #tests both training classes on a test document 
    for c in trainClasses:
            total[i] = logPrior[i]
            for word in testDoc:
                if word in unique:
                    total[i] += loglikehood[i][word]
            i += 1
    #determines if cancer or not cancer based on prob values
    if total[0] > total[1]:
        result = 'cancer'
    elif total[1] > total[0]:
        result = 'nocancer'
    return result
        
#Used to tokenize the documents and also puts them in each of their classes
def tokenizer(ts):
    #takes in the training/testing set file 
    #print("Current working directory: {0}".format(os.getcwd()))
    data = pd.read_csv(os.getcwd() + '/' + ts)
    vocab = [] #words for docs
    cancer_docs = [] #words for cancer class
    nocancer_docs = [] #words for noncancer class
    #itterates through rows of csv
    for index, row in data.iterrows():
        text = row['text']

        '''Tokenize on characters (including white space, of course)
           : \n \t . , ; : ' " ( ) ? !. 
            Do not include these characters as tokens and lowercase all uppercase characters
            in the resulting tokens.'''
        tokens = re.findall("[\w_-]+", text) #Finds all words and hyphens and underscores
        for w in range(len(tokens)): #lowercases them
            tokens[w] = tokens[w].lower()
        vocab += tokens #total words
        if row['class'] == 'cancer': #determines from the next cell in csv what class
            cancer_docs.append(tokens)
        else :
            nocancer_docs.append(tokens)
    return cancer_docs , nocancer_docs , vocab
    

def Stop_Words(ts):
    #reads the stopwords doc line by line then removes the newline char
    fileList = open("stopwords.txt").readlines()
    stopVocab = []
    #goes through the list to find stop words
    for l in fileList:
        stopVocab.append(l.strip())
        #takes in the training set 
    #print("Current working directory: {0}".format(os.getcwd()))
    data = pd.read_csv(os.getcwd() + '/' + ts)
    vocab = []
    cancer_words = []
    nocancer_words = []
    #itterates through rows of csv
    for index, row in data.iterrows():
        text = row['text']
        '''Tokenize on characters (including white space, of course)
           : \n \t . , ; : ' " ( ) ? !. 
            Do not include these characters as tokens and lowercase all uppercase characters
            in the resulting tokens.'''
        tokens = re.findall("[\w'_-]+", text) #Finds all words and hyphens and underscores
        for w in range(len(tokens)):
            tokens[w] = tokens[w].lower()
        #checks the tokens for stopwrods and removes them
        for word in list(tokens):
                if word in stopVocab:
                     tokens.remove(word)
        vocab += tokens #total words
        if row['class'] == 'cancer':
            cancer_words.append(tokens)
        else :
            nocancer_words.append(tokens)
        
    return cancer_words, nocancer_words , vocab


def binary(ts):
    
    data = pd.read_csv(os.getcwd() + '/' + ts)
    vocab = []
    cancer_words = []
    nocancer_words = []
    #itterates through rows
    for index, row in data.iterrows():
        text = row['text']
        '''Tokenize on characters (including white space, of course)
           : \n \t . , ; : ' " ( ) ? !. 
            Do not include these characters as tokens and lowercase all uppercase characters
            in the resulting tokens.'''
        tokens = re.findall("[\w_-]+", text) #Finds all words
        for w in range(len(tokens)):
            tokens[w] = tokens[w].lower()
        unique = [] #makes sure occurances of a word only happens once for a document
        for word in tokens:
            if word not in unique:
                unique.append(word)
        tokens = unique
        vocab += tokens #total words
        if row['class'] == 'cancer':
            cancer_words.append(tokens)
        else :
            nocancer_words.append(tokens)

    return cancer_words, nocancer_words , vocab

def binStop(ts):
    #reads the stopwords doc line by line then removes the newline char
    fileList = open("stopwords.txt").readlines()
    stopVocab = []
    for l in fileList:
        stopVocab.append(l.strip())
        #takes in the training set 
    #print("Current working directory: {0}".format(os.getcwd()))
    data = pd.read_csv(os.getcwd() + '/' + ts)
    vocab = []
    cancer_words = []
    nocancer_words = []
    #itterates through rows
    for index, row in data.iterrows():
        text = row['text']
        '''Tokenize on characters (including white space, of course)
           : \n \t . , ; : ' " ( ) ? !. 
            Do not include these characters as tokens and lowercase all uppercase characters
            in the resulting tokens.'''
        tokens = re.findall("[\w_-]+", text) #Finds all words
        for w in range(len(tokens)):
            tokens[w] = tokens[w].lower()
        #checks the tokens for stopwrods and removes them
        for word in list(tokens):
                if word in stopVocab:
                     tokens.remove(word)
        unique = [] #makes sure occurances of a word only happens once for a document
        for word in tokens:
            if word not in unique:
                unique.append(word)
        tokens = unique
        vocab += tokens #total words
        if row['class'] == 'cancer':
            cancer_words.append(tokens)
        else :
            nocancer_words.append(tokens)
        
    return cancer_words, nocancer_words , vocab

