
<h1>
Naivebayes text classifcation model for cancer and noncancer documents
</h1>
<h3>
  Author: Alex King
</h3>

----------------------------------------------------------------------
1. Purpose
2. Requirements/files included
3. How to use
----------------------------------------------------------------------

<h3>
1. Purpose
</h3>
The Purpose of this program is to read in from csv files containing two columns:
                        
                        Document | classifcation
                        xxxxxx   | cancer/nocancer
                        xxxxxx   | cancer/nocancer
                        xxxxxx   | cancer/nocancer

This program uses the data to read into classes containing each documents one file 
is used as the training set, and the other as the testing set. Each set goes through
the same tokenization. From there one is trained and the other is tested. 
<h3>
2. Requirements/files used
</h3>
* python3 
* numpy library - for calculating log
* pandas library - for reading in csv files
* main.py and naivesbayes.py
* stopwords.txt - list of stop words
* Scoring.docx -  list of scoring for precsion, Recall, F-score
<h3>
3. How to use
</h3>
This program has 3 modes of operation for tokenizing your sets:

                    $python3 main.py -train 1 -test 1 

This first command will execute std tokenization on training set 1 and test set 1. To change 
which training set just change the 1 into a 2.

                    $python3 main.py -train 2 -test 1 

#NOTE do not change testing set number leave it as 1 it was intended for multiple testing sets

For binary:

                    $python3 main.py -train # -test 1 -b

For stopwords:
                    
                    $python3 main.py -train # -test 1 -s
                
For both stopwords and binary:
                    
                    $python3 main.py -train # -test 1 -b -s
                    
