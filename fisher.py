import re
import math
from random import randint
import numpy as np
import sys
from time import time
#import pylab as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


target_names =['Sports','Politics']
tweets=[]
y_train = []
validation_set = []
label_ids=[]
predicted=[]
stopwords={}

class twitminer():	# Hierarchical Tweets Classifier 
	def __init__(self):
		self.feature_extractor = self.word_splitter

	def train(self,cl):
		#sports.txt is extracted from traning.txt with sports tweets
		sl = open("sports.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Sports")
			tweets.append(sl[i])
			y_train.append(0)
		#politics.txt is extracted from traning.txt with politics tweets
		sl = open("politics.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Politics")
			tweets.append(sl[i])
			y_train.append(1)
		#sports_validated.txt is extracted from traning.txt with politics tweets
		sl = open("sports_validated.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Sports")
			tweets.append(sl[i])
			y_train.append(0)
		sl = open("politics_validated.txt","r").read()
		sl = sl.split("\n")
		for i in range(len(sl)):
			cl.train(sl[i],"Politics")
			tweets.append(sl[i])
			y_train.append(1)
			
	def word_splitter(self,doc):
		splitter=re.compile('\\W*')
		words=[s.lower( ) for s in splitter.split(doc) if len(s)>0 and len(s)<20]
		return dict([(w,1) for w in words])
			
	def test_sports(self,cl):
		# Calculates measure of classification,before calling this make sure you trained with only 2000 Sports tweets
		sl = open("sports.txt","r").read()
		sl = sl.split("\n")
		count = 0
		for i in range(2000,3000):
			resu = cl.classify(sl[i])
			if(resu =="Sports"):
				count+=1
		print float(float(count)/1000.0)*100  
	
	def test_politics(self,cl):
		# Calculates measure of classification,before calling this make sure you trained with only 2000 Politics tweets
		sl = open("politics.txt","r").read()
		sl = sl.split("\n")
		count = 0
		for i in range(2000,3000):
			resu = cl.classify(sl[i])
			if(resu =="Politics"):
				count+=1
		print float(float(count)/1000.0)*100

	def addstopwords(self,cl):
		#appends stopwords which were collected from external sources
		f3 =open("stopwords.txt","r").read()
		f3 = f3.split(",")
		for f in f3:
			stopwords[f]=1
	
			
	def start(self):
		#initiates model and starts classification task
		nbcount=0
		svmcount=0
		
		cl1=fisherNB(self.feature_extractor)
		print "Loading Stopwords"
		self.addstopwords(cl1)
		print tweets
	
		print "Training model with Training Data"
		self.train(cl1)
		self.train(cl1)
		nbcount2= self.findresult(cl1,nbcount)
		nbcount2= self.findresult(cl1,nbcount)
		print "Starting Hierarchy 1 Classification"
		nbcount= self.findresult(cl1,nbcount)
		print "Naive bayes classification completed for "+str(nbcount)+" Tweets"
		cl2 = svm(svmcount)
		svmcount = cl2.classify()
		print "Starting Hierarchy 2 Classification"
		print "SVM classification completed for remaining "+str(svmcount)+" Tweets"
		print "Result file for "+str(nbcount+svmcount)+" stored in result.txt"

	def findresult(self,cl1,nbcount):
		f = open("test.txt","r").read()
		f = f.split("\n")
		count = 0
		ucount=0
		f3 = open("result.txt","w")
		f4 = open("leftforsvm.txt","w")
		for wa in range(len(f)):
			wx = re.findall(r'\d+',f[wa])
			for i in range(len(wx)):
				if(int(wx[i])>1750962595):
					resu = cl1.classify(f[wa])
					if(resu!=""):
						f3.write(wx[i]+" "+resu+"\n")
						nbcount+=1
					else:
						f4.write(f[wa]+"\n")
		f3.close()
		f4.close()
		return nbcount
	
class svm(): # Hierarchy-II Classifier
	def	__init__(self,svmcount):
		self.svmcount = 0
		self.train()
	def train(self):
		self.X_train = np.array(tweets)
	def classify(self):
		f = open("leftforsvm.txt","r").read()
		f = f.split("\n")
		for wa in range(len(f)):
			wx = re.findall(r'\d+',f[wa])
			for i in range(len(wx)):
				if(int(wx[i])>1750962595):
					validation_set.append(f[wa])
					label_ids.append(wx[i])
		X_test = np.array(validation_set)   
		classifier = Pipeline([
			('vectorizer', TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')),
			('tfidf', TfidfTransformer()),
			('clf',  LinearSVC())])
		classifier.fit(self.X_train, y_train)
		predicted = classifier.predict(X_test)
		t2count=0
		f3 = open("result.txt","a")
		for ids,labels,tweet in zip(label_ids,predicted,validation_set):
			f3.write(ids+" "+target_names[labels]+"\n")
			self.svmcount+=1
		f3.close()
		return self.svmcount

class fisherNB():	# Hierarchy-I Classifier
	def __init__(self,getfeatures,filename=None):
		self.featurecount={}
		self.categorycount={}
		self.getfeatures=getfeatures
		self.minimums={}

	def feature_count(self,f,cat):
		if f in self.featurecount and cat in self.featurecount[f]:
			return float(self.featurecount[f][cat])
		return 0.0
		
	def catcount(self,cat):
		if cat in self.categorycount:
			return float(self.categorycount[cat])
		return 0
		
	def train(self,item,cat):
		features=self.getfeatures(item)
		for f in features:
			if f in stopwords:
				continue
			else:
				self.featurecount.setdefault(f,{})
				self.featurecount[f].setdefault(cat,0)
				self.featurecount[f][cat]+=1
				self.categorycount.setdefault(cat,0)
				self.categorycount[cat]+=1
				
	def fprob(self,f,cat):
		if self.catcount(cat)==0: return 0
		return self.feature_count(f,cat)/self.catcount(cat)
		
	def cprob(self,f,cat):
		clf=self.fprob(f,cat)
		if clf==0: return 0
		freqsum=sum([self.fprob(f,c) for c in self.categorycount.keys()])	# total frequency of feature
		p=clf/(freqsum)	
		return p
		
	def fisherprob(self,item,cat):	#calculates fisher probability
		p=1
		features=self.getfeatures(item)
		for f in features:
			basicprob=self.cprob(f,cat)
			totals=sum([self.feature_count(f,c) for c in self.categorycount.keys()])
			p *= ((1.0*0.5)+(totals*basicprob))/(1.0+totals)
			fscore=-2*math.log(p)
		return self.invchi2(fscore,len(features)*2)	# Fit into Inverse Chi 2 function
		
	def invchi2(self,chi,df):
		m = chi / 2.0
		sum = term = math.exp(-m)
		for i in range(1, df//2):
			term *= m / i
			sum += term
		return min(sum, 1.0)		
			
	def classify(self,item,default=None):
		best=default
		max=0.0
		for c in self.categorycount.keys():
			p=self.fisherprob(item,c)
			if c not in self.minimums:
				minimum = 0
			else:
				minimum = self.minimums[c]
			if (p>minimum and p>max):
				best=c
				max=p
		p1 = self.fisherprob(item,"Sports")
		p2 = self.fisherprob(item,"Politics")
		p3 = abs(p1-p2)*10
		p3 = int(p3)
		maxcount=0.0
		if(p3<5.0):	 # Calls Heirarchy II
			best=''	 # These emptied predictions will be forwarded to SVM
		else:
			for za in range(p3/2):
				self.train(item,best)	# On-line learning of model
			tweets.append(item)
			if(best=="Sports"):
				y_train.append(0)
			else:
				y_train.append(1)
		return best

def start_twitminer():
	classifier = twitminer()
	classifier.start()
	
if  __name__ =='__main__':start_twitminer()
