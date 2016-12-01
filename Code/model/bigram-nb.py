import pandas as p
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import json
import math

tokens = '../Data/tokenized_data.json'

filename = "../Data/x_vector/Y.csv"
y_read = p.read_csv(filename)
y_data = np.array(y_read[1:4001])

[n,l] = y_data.shape
n_classes = l
print n, l
f = open(tokens,'r')
data = json.load(f)

testrows = []
trainrows = []
trainy = []
testy = []
max_f_score = -1.0
tuned_C = -1
tuned_loss = ''
tuned_penalty = ''

all_labels = ["javascript", "php", "android", "jquery", "java", "asp.net", "c++","iphone", "python", "c"]

data = data[1:]
count = 0
#print data[0]
#print len(data)
for row in data:
	if count < 3000:
		trainrows.append(' '.join(word for word in row["title"]))
	elif count < 4000:
		testrows.append(' '.join(word for word in row["title"]))
	else:
		break
	count += 1

X_train = trainrows
X_test = testrows		

tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2), stop_words = 'english')
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

model_nb_tfidf = BernoulliNB(alpha=0.00001)
#model_nb_tfidf = MultinomialNB(alpha=0.00001, fit_prior=True, class_prior=None)
#model_nb_tfidf = GaussianNB()
split = 3000
for i in range(10):
	Y_train = y_data[:split,i]
	#print Y_train
	y_test = y_data[split:,i]
	#print y_test
	model_nb_tfidf.fit(X_train_tfidf, Y_train)
	y_pred = model_nb_tfidf.predict(X_test_tfidf)
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	for j in range(len(y_pred)):
		if y_test[j]==1 and y_pred[j]==1:
		    tp+=1
		if y_test[j]==-1 and y_pred[j]==1:
		    fp+=1
		if y_test[j]==1 and y_pred[j]==-1:
		    fn+=1
		if y_test[j]==-1 and y_pred[j]==-1:
		    tn+=1
	    
	"""      
	print("True Postives:",tp)
	print("False Postives:",fp)
	print("False Negatives:",fn)
	print("True Negatives:",tn)
	"""
	accuracy = (tp+tn)*1.0/(tp+tn+fp+fn)
	error = (fp+fn)*1.0/(tp+tn+fp+fn)
	recall = tp*1.0/(tp+fn)
	if tp==0 and fp==0:
		precision = 0
	else:
		precision = tp*1.0/(tp+fp)


#	print("Train Accuracy:",clf.score(X_train,y_train))
	print("Test Accuracy:",accuracy)
	print("Test Error:",error)
	print("Recall:",recall)
	print("Precision:",precision)
	#print("Specificity:",specificity)

	print ("***************************************")

	if recall==0 and precision==0:
		f1_score = 0
	else:
		f1_score = 2.0*recall*precision/(recall+precision)

	if(f1_score>max_f_score):
		max_f_score = f1_score

print("Tuned F1 Score",max_f_score)
