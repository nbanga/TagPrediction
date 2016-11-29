import json
import math

tokens = '../Data/tokenized_data.json'

f = open(tokens,'r')
data = json.load(f)

#generate count - dictionary of dictionaries. count[word][tag] stores the count of entries with word and tag in the corpus
testrows = []
exptags = []
alltags = []

count = {}
entry = 0
for row in data:
    if entry < 1:
        testrows.append(row["title"])
        exptags.append(row["tags"])
        entry += 1
        alltags.extend(row["tags"])
    else:
        break

data = data[1:]

for row in data:
    title = row["title"]
    alltags.extend(row["tags"])
    tags = row["tags"]
    for word in title:
        for tag in tags:
            if word not in count.keys():
                count[word] = {}
                count [word][tag] = 1
            else:
                if tag not in count[word].keys():
                    count[word][tag] = 1
                else:
                    count[word][tag] += 1

                    #print count

#get count of occurances of a tag across documents
docCount = {}
for row in data:
    tags = row["tags"]
    for tag in tags:
        if tag not in docCount.keys():
            docCount[tag] = 1
        else:
            docCount[tag] += 1


#doc = "file open C unable help"
#xspredict(doc)
def Pr(word, tag):
    numerator = 1
    if word in count.keys():
        if tag in count[word].keys():
            numerator += count[word][tag]

    denominator = 0;
    V = len(count.keys())
    for word in count.keys():
        if tag in count[word].keys():
            denominator += count[word][tag]
    return numerator*1.0/(denominator + V)

def Pr1(tag):
    totalDocs = len(data)
    return docCount[tag]*1.0/totalDocs

def bagofwords(doc):
    return doc

def perf(exp, pred):
    tp = len(exp.intersection(pred))
    fn = len(pred.difference(exp))
    fp = len(exp.difference(pred))
    tn = len((set(alltags)).difference(exp.union(pred)))
    return [tp, fn, fp, tn]

def predict(doc):
    e = {}
    doc = bagofwords(doc)
    tags = docCount.keys()
    for tag in tags:
        e[tag] = math.log(Pr1(tag))
        prod = 0.0
        for word in doc:
            prod += math.log(Pr(word, tag))
        e[tag] += prod

    #    getPerf()
    #get key with max value in dictionary
    predictedTags = sorted(e, key=e.__getitem__)
    print predictedTags[-3:]
    return predictedTags[-3:]

TP = 0
FN = 0
FP = 0
TN = 0
for i in range(len(testrows)):
    print exptags[i]
    acttags = predict(testrows[i])
    [tp, fn, fp, tn] = perf(set(exptags[i]), set(acttags))
    TP += tp
    FN += fn
    FP += fp
    TN += tn
    print "---"

print ("Performance summary")
print ("Accuracy : ", 1.0*(TP+TN)/(TP+FP+FN+TN))
print ("Error : ", 1.0*(FP+FN)/(TP+FP+FN+TN))
print ("Recall/Sensitivity : ", 1.0*TP/(TP+FN))
print ("Precision : ", 1.0*TP/(TP+FP))
print ("Specificity : ", 1.0*TN/(TN+FP))
print ("*********DONE**********")
