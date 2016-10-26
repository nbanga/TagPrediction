import re

parsedPath = "../../Data/parsed_data.json"
prunedPath = "../../Data/pruned_data.csv"

def data_prune(parsedPath):
    print("Removing entries with all tag words in question/answer tokens")

    f = open(prunedPath, 'w')
    with open(parsedPath,'r') as corpus:
        for line in corpus:


