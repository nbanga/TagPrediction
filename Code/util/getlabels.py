import json
import operator
import os

#input file
filepath = "../../Data/tokenized"

#output file
lpath = "../../Data/all_labels.json"
max_index = 10

fw = open(lpath,'w')
labels = dict()

def getLabels(file):
    print("In getLabels ...")
    fin = open(file,'r')
    data = json.load(fin)

    for row in data:
        for each in row["tags"]:
            if labels.has_key(each):
                labels[each] = labels[each]+1
            else:
                labels.setdefault(each,1)

    sorted_labels = sorted(labels.items(), key = operator.itemgetter(1), reverse = True)
    json.dump(sorted_labels[0:max_index], fw)
    print("Exit getLabels ...")

def main():
    for dir, subdir, files in os.walk(filepath):
        for filename in files:
            f = os.path.join(dir,filename)
            getLabels(f)


if __name__ == '__main__':
    main()
