import json
import os,sys

filepath = "../../Data/x_vector"
tokenpath = "../../Data/tokenized_data.json"
path = "../../Data/all_labels.json"

def split_by_label(label,num_rows):
    label_filepath = filepath + "/" + label + ".json"
    feed = []
    curr_file = open(tokenpath,'r')
    data = json.load(curr_file)
    for row in data:
        if label in row["tags"]:
            feed.append(row)
            if len(feed)==num_rows:
                break

    #print(feed)
    label_file = open(label_filepath,'w')
    json.dump(feed,label_file)
    label_file.close()

def main():
    file = open(path,'r')
    data = json.load(file)
    for label in data:
        print label
        split_by_label(label[0][0],100)

if __name__=='__main__':
    main()
