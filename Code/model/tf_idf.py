from sklearn.feature_extraction.text import TfidfVectorizer
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
import numpy
import math
import csv


path = '../../Data/x_vector'
labels = '../../Data/all_labels.json'
tokens = '../../Data/tokenized_data.json'


#for subdir, dirs, files in os.walk(path):
#    for file in files:
        #file_path = subdir + os.path.sep + file

word_dict = {}
X = []

file = open(tokens,'r')
data = json.load(file)
file.close()

index = 0
for row in data:
    #x = {'freq': {}, 'tf': {}, 'idf': {}, 'tf-idf': {}, 'tokens': []}
    s = set(row["title"])
    for title in s:
        if title not in word_dict.keys():
            word_dict[title] = [index, 1]
            index += 1
        else :
            word_dict[title][1] = word_dict[title][1] + 1

d = len(data)
l = len(word_dict)

word_dict_tf = dict(map(lambda (k,(u,v)): (k, (u, math.log(d/float(v)))), word_dict.iteritems()))

X = [[0.0]*l]*d
for i in range(d):
    p = data[i]["title"]
    temp = [0.0]*l
    for word in p:
        temp[word_dict_tf[word][0]] = word_dict_tf[word][1]*(1.0*p.count(word)/len(p))
    X[i] = temp

label_file = open(labels,'r')
label = json.load(label_file)

Y = [[-1]*100]*d
for i in range(d):
    p = data[i]["tags"]
    temp = [-1]*100
    for word in p:
        for sublist in label:
            if sublist[0] == word:
                temp[label.index(sublist)] = 1
                break
    Y[i] = temp

label_file.close()

with open(path + "/X.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X)
f.close()

with open(path + "/Y.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Y)
f.close()








