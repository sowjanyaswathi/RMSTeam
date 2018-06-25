# Submission code
          --RMSTeam
-----------------------------------------------------------------

This is the working code for classification done by using past behaviour, which we implemented for  machine learning contest.
To run this code after installing sklearn, please type this command in your terminal
>	python fisher.py

------Requirements:----------------------

1. Python sklearn package version 0.13.1

You can install scikit-learn-0.13.1 scikit-learn.org

sudo apt-get install build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev

-----Files contained in this code:--------

This code contains following files

1."fisher.py" 
	Complete implementation code contains in this file. 
	It uses two classification algorithms 
	i) Fisher score based Naive bayesian Classification
	ii) Linear SVM text classifier
2. "politics.txt"
	This politics.txt file only contains Politics tweets extracted from "training.txt", that you have provided before
3. "sports".txt"
	This sports.txt file only contains Sports tweets extracted from "training.txt", that you have provided before
4. "politics_validated.txt"
	It contains validated Politics tweets which are nothing but ouput prediction of my model for validaion data you provided.
	It only contains tweet predictions with high decision probability. 
	I extracted predictions with more than 0.5 diffrence probability i.e Pr(Sports/tweet)-Pr(Politics/tweet)
5. "sports_validated.txt"
	It contains validated Sports tweets which are predicted by my model during validation phase with high probability difference 
	which is explained earlier.
6. "stopwords.txt"
	This is collection of some stop words which were gathered from external sources. 
Note: I have not used any other sports or politics external keywords.
7. "validation.txt" and "test.txt"
8. "result.txt"
	Output predictions of my model will be stored in this file.

This code will create "leftforsvm.txt" which is used by svm in validation algorithm.

Note: This code does not contain any other external datasets or keywords except some few stopwords. 

	All above tweets used as training set are extracted only from "training.txt"(6526) and "validation.txt"(2609) files you have provided earlier.
