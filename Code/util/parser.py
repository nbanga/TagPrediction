import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

filepath = "../../Data/data/Train_21.csv"
parsedPath = "../../Data/parsed_data.json"

def bigparse(filepath):
    print("In bigparse ... ")
    pattern = r'"([0-9]+)"'
    question = ""
    f = open(parsedPath, 'w')
    with open(filepath,'r') as corpus:
        for line in corpus:
            tmp_line = line.replace('""','<empty>')
            found = re.match(pattern, tmp_line)
            if(found):
                # process the question
                row = question.split('","')
                #print(row)
                for i in range(len(row)):
                    tokenized_row = word_tokenize(row[i])
                    #row2 = word_tokenize(row[2])
                    #row3 = word_tokenize(row[3])
                    rowtemp = []
                    for w in tokenized_row:
                        if w.lower() not in stop_words:
                            rowtemp.append(w.lower())
                    row[i] = rowtemp
                    #print(each)
                f.write(str(row))
                
                #print()
                #print(word_tokenize(row[2]))
                #print(word_tokenize(row[-1]))
                #print(row[0].strip('"') + " : ",)
                #print(row[-1].replace('"\r\n',"").split(' '))
                question = line
            else:
                #add to existing question
                question = question+line

    f.close()

bigparse(filepath)
