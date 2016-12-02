# TagPrediction

This is a tag prediction software which implements multiple ML algorithms using Bag of Words and TF-IDF to train data.
This Code is compatible with Python 2.7

#Prerequisites:

pip install sklearn
pip install scipy
pip install pandas
pip install numpy
pip install nltk
pip install BeautifulSoup4
pip install python-matplotlib
Got to your python prompt and run nltk.download()

Modify nltk’s tokenize file treebank.py to remove ‘#’ and add ‘=’ to the regex of delimiters.
In line 60:

   PUNCTUATION = [
       (re.compile(r'([:,])([^\d])'), r' \1 \2'),
       (re.compile(r'([:,])$'), r' \1 '),
       (re.compile(r'\.\.\.'), r' ... '),
       (re.compile(r'[;@=$%&]'), r' \g<0> '),
       (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
       (re.compile(r'[?!]'), r' \g<0> '),

       (re.compile(r"([^'])' "), r"\1 ' "),
   ]

#Run
run ./tagPrediction.sh


