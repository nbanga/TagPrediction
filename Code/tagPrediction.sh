#!/bin/bash

#split Train.csv into intermediate files
# Shell commands differ between OSes

##############################
# Uncomment for regenerating csv data files
##############################
#rm ../Data/data/train*.csv
#split --bytes 1M --numeric-suffixes --suffix-length=4 --additional-suffix=".csv" ../Data/data/Train.csv ../Data/data/train
#
#for f in ../Data/data/train*.csv; do
##    echo "$f"
#    sed -i '1i Id,Title,Body,Tags' $f
#done
echo 'Split done.'
##############################
# Uncomment till here
##############################

#Call parser on each intermediate file to generate intermediatexy.csv
#Call tokenizer on each intermediatexy.csv generate tokxy.csv

##############################
# Uncomment for pre-processing data
##############################
#echo "Create Tokenized files"
#python util/parse_csv.py
#
#
##Create map of labels and frequency
#echo "Collect top tags"
#python util/getlabels.py
#
#
## Get 500 data points for top 10 labels
#echo "Collect 500 samples for top 10 tags"
#python util/split_by_label.py
#
#
##Create X and Y vectors
#echo "Create X and Y vectors"
#python model/tf_idf.py
##############################
# Uncomment till here
##############################

# Chose family of classifier for svm
echo "Run svm with linear, rbf and 3rd order poly"
python model/chose_family.py

# Call linear SVM on the vectorized data
echo "Tune linear svm on vectorized data without feature selection"
python model/linear_svm_tuning.py

# Chose family of classifier for svm
echo "Data visualization, k-fold crossvalidation with PCA using linear SVM"
python model/svm_pca.py

# Call Random forests on vectorized data
echo "Run Multivariate and single variate Random forests on vectorized data"
python model/decision_tree.py

# Call naive Bayes on the tokenized data with bigrams
echo "Applying Naive Bayes with bigrams to tokenized file"
python model/bigram-nb.py

# Call naive Bayes on the tokenized data
echo "Applying multiplicative Naive Bayes to tokenized file"
python model/linear_nb.py

# Call naive Bayes on the tokenized data
echo "Applying Naive Bayes to tokenized file"
python model/nb.py

#Algorithm complete.
echo "Done"
