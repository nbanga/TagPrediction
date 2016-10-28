import json
import os,sys

filepath = "../../Data/x_vector"
tokenpath = "../../Data/tokenized"

def split_by_label(label,num_rows):
    label_filepath = filepath + "/" + label + ".json"
    feed = []
    for file in os.listdir(tokenpath):
        curr_file = open(file,'r')
        data = json.load(curr_file)
        for row in data:
            if label in row["tags"]:
                feed.append(row)
                if len(feed)==num_rows:
                    break
        if len(feed)==num_rows:
            break

    label_file = open(label_filepath,'w')
    json.dumps(feed,label_file)

def main():
    split_by_label(label,num_rows)

if __name__=='__main__':
    main()
