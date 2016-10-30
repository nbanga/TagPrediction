from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

filepath = "../../Data/all_labels.json"

label_file = open(filepath,'r')
labels = "" + label_file.read()
tag_list = (pos_tag(word_tokenize(labels)))
count = 0
labels = {}
for each in tag_list:
    #print (each)
    #if each[1] not in ['NN', 'NNS', 'VBG','VBN','VBD','VBZ','VBP','RB','VB', 'JJ','JJS','JJR','#','VBD',')','CD','.',';',':',',','``',"''"]:
    #print(each)
    if labels.get(each[1]) == None:
        labels[each[1]] = 1
    else:
        labels[each[1]] = labels[each[1]]+1;

#print(labels)
label_file.close()