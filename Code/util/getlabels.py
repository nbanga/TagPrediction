import json
import operator

#input file
filepath = "../../Data/tokenized_data.json"

#output file
lpath = "../../Data/all_labels.json"
max_index = 10

fw = open(lpath,'w')
labels = dict()

def getLabels():
    print("In getLabels ...")
    fin = open(filepath,'r')
    data = json.load(fin)
    
    for row in data:
        for each in row["tags"]:
            if labels.has_key(each):
                labels[each] = labels[each]+1
            else:
                labels.setdefault(each,1)

    sorted_labels = sorted(labels.items(), key = operator.itemgetter(1), reverse = True)
    json.dump(sorted_labels[:max_index], fw)
    print("Exit getLabels ...")

def main():
    getLabels()


if __name__ == '__main__':
    main()
