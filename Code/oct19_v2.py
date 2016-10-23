import re

filepath = "C:\\Users\\Archit\\OneDrive - purdue.edu\\Data\\Courses\\CS578 ML\\Project\\Train.csv"

def bigparse(filepath):
    print("In bigparse ... ")
    pattern = r'"([0-9]+)"'
    question = ""
    with open(filepath,'r') as corpus:
        for line in corpus:
            tmp_line = line.replace('""','<empty>')
            found = re.match(pattern, tmp_line)
            if(found):
                # process the question
                row = question.split('","');
                print(row[0].strip('"') + " : ",)
                print(row[-1].replace('"\r\n',"").split(' '))
                question = line
            else:
                #add to existing question
                question = question+line


bigparse(filepath)


#Traceback (most recent call last):
#  File "C:/Users/Archit/OneDrive - purdue.edu/Data/Courses/CS578 ML/Project/229-224N-Project-feature-pipeline/oct19.py", line 61, in <module>
#    bigparse(filepath)
#  File "C:/Users/Archit/OneDrive - purdue.edu/Data/Courses/CS578 ML/Project/229-224N-Project-feature-pipeline/oct19.py", line 45, in bigparse
#    for line in corpus:
#  File "C:\Users\Archit\AppData\Local\Programs\Python\Python35\lib\encodings\cp1252.py", line 23, in decode
#    return codecs.charmap_decode(input,self.errors,decoding_table)[0]
#UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 228: character maps to <undefined>