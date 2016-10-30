#!/bin/bash

#split Train.csv into intermediate files
rm ../Data/data/train*.csv
sh ./splitTrain.sh
echo 'Split done.'

#Call parser on each intermediate file to generate intermediatexy.csv
#Call tokenizer on each intermediatexy.csv generate tokxy.csv
python util/parse_csv.py

#Create map of labels and frequency
#python getlabels.py

#Collect N samples for each label


#Data processing is done.
echo Data processing is done!

#Call data modelling module
