import os, sys
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import model.svm_common as com

filename = '../../Data/Uniform/X.csv'
X_read = pd.read_csv(filename)

filename = '../../Data/Uniform/Y.csv'
Y_read = pd.read_csv(filename)

X = np.array(X_read)
Y = np.array(Y_read)

"""
# Step 1
# Visualize data in 2D using TSNE and PCA Libraries
destination = '../../Data/Plots/Visualize/plotTSNE.pdf'
com.plotTSNE(np.array(X_read), np.array(Y_read), destination)
"""

# Step 2
# Visualize data in 2D using PCA written by hand
# F is the optimum value of # of params containing 95% of information
F = com.getplotsPCA(np.array(X_read), np.array(Y_read))


"""
# Step 3
# Customize C to train different models
C = [0.1,0.5,1,10,0.05,0.01,0.005,0.001,0.0005]
loss = ['hinge','squared_hinge']
# Set number of folds
k = 10
# Set class number to do the tuning for (1 to 10, we did for 1)
c = 1
f, z, mu, var = com.tuneModel(X, Y, C, loss, k, c)
"""

"""
# Step 4
# Plot number of optimum features for each fold
destination = '../../Data/Plots/SVM/plotNumFeat.pdf'
com.plotNumFeat(f, destination)
"""

"""
# Step 5
# Plot average F1 score for each experiment
destination = '../../Data/Plots/SVM/plotAvgF1.pdf'
mu2, var2 = com.plotAvgF1(z, mu, destination)
"""

"""
# Step 6
# Get t-statistic and degree of freedom for two models to compare
import math
m1 = 0.527594325
m2 = 0.533902935
var1 = 0.003700582
var2 = 0.001801053
n = X.shape[0]
x = (m1-m2)*math.sqrt(n)/(math.sqrt(var1+var2))
df = math.pow((var1+var2),2)*(n-1)/(math.pow(var1,2)+math.pow(var2,2))
"""