import re
import csv
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

stop_words = set(stopwords.words('english'))

filepath = "../../Data/data/Train_21.csv"
parsedPath = "../../Data/parsed_data.json"
tokenizedPath = "../../Data/tokenized_data.json"

def read_csv_to_dict(filepath, parsedPath):

    # read from csv into json format
    inputCsv = open(filepath, 'rU')
    reader = csv.DictReader(inputCsv)
    out = json.dumps([row for row in reader])

    # remove code blocks
    while 1:
        if "<code>" in out:
            start = out.find("<code>")
            end = out.find("</code>") + 7
            out = out.replace(out[start:end], "")
        else:
            break

    # remove html tags and http links
    pattern = r'(<.*?>)'
    out = re.sub(pattern,"",out)
    http_pattern = r'(http.*?\/\/.*?\r?\n)'
    out = re.sub(http_pattern,"",out)

    out = out.lower()
    #decode('unicode_escape').encode('ascii','ignore').

    # write data to json file
    outputJson = open(parsedPath,'w')
    outputJson.write(out)
    outputJson.close()

def tokenize_data(parsedPath, tokenizedPath):
    parsedFile = open(parsedPath,'r')
    tokenizedFile = open(tokenizedPath,'w')
    data = json.load(parsedFile)

    feeds = []
    for row in data:
        row["tags"] = word_tokenize(row["tags"])

        row["body"] = [word for word in word_tokenize(row["body"]) if word not in stopwords.words("english")]
        for each in pos_tag(row["body"]):
            if each[1] not in ['NN', 'JJ', 'NNS', 'VBG']:
               row["body"].remove(each[0])

        title = [word for word in word_tokenize(row["title"]) if word not in stopwords.words("english")]
        row["title"] = title
        for each in pos_tag(row["title"]):
            if each[1] not in ['NN', 'JJ', 'NNS', 'VBG']:
                row["title"].remove(each[0])

        feeds.append(row)

    json.dump(feeds, tokenizedFile)

def main():
    read_csv_to_dict(filepath,parsedPath)
    tokenize_data(parsedPath, tokenizedPath)

if __name__=='__main__':
    main()
