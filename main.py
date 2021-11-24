#Alex King 2021
#Naive Bayes text classifcation for cancer and noncancer documents


import sys #used for files
import naivebayes as nb #file contains our code for naivebayes functions

if __name__ == "__main__":
    #specfiy training doc
    if sys.argv[1] == "-train":
        if sys.argv[2] == "1" :
            file = 'training1_v1.csv'
        elif sys.argv[2] == "2":
            file = 'training2_v1.csv' 
    else:
        print("ERROR MISSING TRAINING SET")
    #test file user wants to use
    if sys.argv[3] == "-test":
        if sys.argv[4] == "1":
            test = 'testing1_v1.csv'
        if sys.argv[4] == "2":
            test = 'testing2_v1.csv'
    else:
        print("Error Missing test set")

    print("Training set used: ", file)
    print("Test set used: ", test) 
    #then calls the standard naive bayes method if argument count is lower than 
    if len(sys.argv) == 5:
        #tokenize the values
        canc_train , noncanc_train , vocab = nb.tokenizer(file) #training set variables
        canc_test , noncanc_test, test_vocab = nb.tokenizer(test) #testing set variables
        print("Stopwords included: N")
        print("Binary version: N")

    if len(sys.argv) == 6: #used for stopwords and binary
        if sys.argv[5] == "-s": #just stop words
            canc_train , noncanc_train , vocab = nb.Stop_Words(file) #training set variables
            canc_test , noncanc_test, test_vocab = nb.Stop_Words(test) #testing set variables
            print("Stopwords included: Y")
            print("Binary version: N")

        elif sys.argv[5] == "-b": #just binary
            canc_train , noncanc_train , vocab = nb.binary(file) #training set variables
            canc_test , noncanc_test, test_vocab = nb.binary(test) #testing set variables
            print("Stopwords included: N")
            print("Binary version: Y")

    if len(sys.argv) == 7: #Binary and stopwords
        canc_train , noncanc_train , vocab = nb.binStop(file) #training set variables
        canc_test , noncanc_test, test_vocab = nb.binStop(test) #testing set variables
        print("Stopwords included: Y")
        print("Binary version: Y")
   
    #trainingclasses and naivebayes training model functions
    trainClasses = [ canc_train, noncanc_train]
    logPrior , loglikehood , unique = nb.trainnaivebayes(trainClasses,vocab)
    resultsCancerTest = nb.testclass(canc_test, logPrior, loglikehood, trainClasses, unique) #cancerdocstesting
    resultsNoCancerTest =nb.testclass(noncanc_test, logPrior, loglikehood, trainClasses, unique) #noncancerdocstesting

    #lists results for user
    truePos = resultsCancerTest.count('cancer')
    print("TP: ", truePos)
    falsePos = resultsNoCancerTest.count('cancer')
    print("FP: ", falsePos)
    falseNeg = resultsCancerTest.count('nocancer')
    print("FN: ", falseNeg)
    trueNeg = resultsNoCancerTest.count('nocancer')
    print("TN: ", trueNeg)
    precision = truePos/ (truePos + falsePos)
    print("precision: ", precision)
    recall =  truePos/ (truePos + falseNeg)
    print("Recall: ", recall)
    fScore = (2*precision*recall)/ (precision+recall)
    print("Fscore: ", fScore)
    

