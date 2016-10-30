#!/bin/bash
split --bytes 500M --numeric-suffixes --suffix-length=2 --additional-suffix=".csv" ../Data/data/Train.csv ../Data/data/train
