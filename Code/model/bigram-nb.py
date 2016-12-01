import pandas as p
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import math

#read corpus
tokens = '../../Data/tokenized_data.json'
f = open(tokens,'r')
data = json.load(f)

#read labels matrix
filename = "../../Data/x_vector/Y.csv"
y_read = p.read_csv(filename)
y_data = np.array(y_read[1:4001])

[n,l] = y_data.shape
n_classes = l

#initialize global variables
testrows = []
trainrows = []
trainy = []
testy = []
max_f_score = -1.0

#top 10 labels for which classification is done
all_labels = ["javascript", "php", "android", "jquery", "java", "asp.net", "c++","iphone", "python", "c"]

#discount the title row
data = data[1:]

count = 0
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

min_n = 2
max_n = 2
#uncomment the below lines to run unigram Naive Bayes
#min_n = 1
#max_n = 1

tfidf_vectorizer = TfidfVectorizer(ngram_range=(min_n,max_n), stop_words = 'english')
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#comment the below and uncomment MultinomialNB line for running the other model
#model_nb_tfidf = BernoulliNB(alpha=0.00001)
model_nb_tfidf = MultinomialNB(alpha=0.00001, fit_prior=True, class_prior=None)

split = 3000
for i in range(10):
	Y_train = y_data[:split,i]
	y_test = y_data[split:,i]
	model_nb_tfidf.fit(X_train_tfidf, Y_train)
	y_pred = model_nb_tfidf.predict(X_test_tfidf)
	tp = fp = fn = tn = 0
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

#	print("Test Accuracy:",accuracy)
#	print("Test Error:",error)
#	print("Recall:",recall)
#	print("Precision:",precision)

	if recall==0 and precision==0:
		f1_score = 0
	else:
		f1_score = 2.0*recall*precision/(recall+precision)

	if(f1_score>max_f_score):
		max_f_score = f1_score

	print(all_labels[i], ": F1 Score",f1_score)
