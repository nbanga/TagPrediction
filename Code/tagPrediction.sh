#!/bin/bash

#split Train.csv into intermediate files
# Shell commands differ between OSes
rm ../Data/data/train*.csv
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    split --bytes 500M --numeric-suffixes --suffix-length=2 --additional-suffix=".csv" ../Data/data/Train.csv ../Data/data/train
elif [[ "$OSTYPE" == "darwin"* ]]; then
    split -b 5m -a 3 ../Data/data/Train.csv ../Data/data/train
fi
echo 'Split done.'

#Call parser on each intermediate file to generate intermediatexy.csv
#Call tokenizer on each intermediatexy.csv generate tokxy.csv
python util/parse_csv.py
echo "Tokenized files created"

#Create map of labels and frequency
python getlabels.py
echo "top tags collected"

# Get 500 data points for top 10 labels
python split_by_label.py
echo "got 500 samples for top 10 tags"

#Create X and Y vectors
python tf-idf.py
echo "formed X and Y vectors"

# Call naive Bayes on the tokenized data
python nb.py
echo "Applied Naive Bayes to tokenized file"

# Call linear SVM on the vectorized data
echo "Applied linear svm to vectorized data"

# Call Radial Basis kernel SVM
echo "Applied RBF svm to vectorized data"

# Call Polynomial Basis kerenl SVM
echo "Applied polynomial svm to vectorized data"

#Algorithm complete.
echo "Done"
