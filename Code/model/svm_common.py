import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


"""
import pylab as pl
from itertools import cycle
def plot_2D(data, target, target_names, destination):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    pl.xlabel('PCA Component #1', fontsize=12)
    pl.ylabel('PCA Component #2', fontsize=12)
    pl.title('Plot of top two PCA components for different classes')
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label)
    pl.legend()
    pl.savefig(destination)
    
#target_names = ['1','2','3','4','5','6','7','8'] #,'8','9'];
#plot_2D(X_proj[y2==1], y3[y2==1], target_names, destination)
"""

"""
Input: Projected matrix X
Multiclass label matrix Y of size #samples x #classes
minind-maxind: range of data to output
Output: x_2: Output data
y_2: Output labels (0 to 9)
"""
def getVisData(X, Y):
    
    y_temp = Y
    # Convert labels in 0-1 format
    y_temp[y_temp==-1] = 0
    # Get sum to know number of classes for each samples
    numclass = np.sum(y_temp,axis=1)
    # Replace 1 by class number
    for i in range(y_temp.shape[0]):
        for j in range(y_temp.shape[1]):
            y_temp[i,j]=y_temp[i,j]*(j+1)
    # Convert Y to a column vector
    actclass = np.sum(y_temp,axis=1)
    # Select only those samples and their label which have just one class
    x_2 = X[numclass==1]
    y_2 = actclass[numclass==1]
    return x_2, y_2
    

"""
Scatter plot x, y group by label and save plot at destination 
(TSNE and PCA Libraries)
"""
def plotTSNE(X, Y, destination):
    
    x_2, y_2 = getVisData(X,Y)
    X_tsne = TSNE(learning_rate=100).fit_transform(x_2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_2)
    plt.title('Data Visualization in 2D using TSNE Library')
    destination = '../Data/Plots/Visualize/plotTSNE.pdf'
    plt.savefig(destination)
    plt.close()
    
    X_pca = PCA().fit_transform(x_2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_2)
    plt.title('Data Visualization in 2D using PCA Library')
    destination = '../Data/Plots/Visualize/plotPCALibrary.pdf'
    plt.savefig(destination)
    plt.close()


"""
Input: Matrix X of M rows and N columns
p: % infomation needed (0 to 1)
Output: Array u of mean of N columns
Matrix Z of learned matrix having p% of information
Array ev of eigen values of size min(M,N)
F: Optimum # of features
"""
def pcalearn(X, p):
    
    # Get # of samples
    n = X.shape[0]
    # Get row vector of mean
    u = np.sum(X,axis=0)/n
    # Helper matrix    
    ones = np.ones([n,1])
    # Center the data
    X_cent = X - ones*u
    # Do SVD
    # U = MxMin(M,N); s = array(Min(M,N)); V = Min(M,N)xN
    U, s, V = np.linalg.svd(X_cent, full_matrices=False)
    # Obtain Eigen Values
    ev = np.square(s)
    # Find # of features have 90% information about data
    F = sum(ev > (1-p)*ev[0])
    # Get top F eigen values
    E = np.diag(np.reciprocal(s[:F]))
    W = V[:F,:].transpose()
    #Reduced features
    Z = math.sqrt(n)*np.dot(W,E)
    return u, Z, ev, F


"""
Input: Array ev of eigen values
destination - path to save plot
Output: Plot saved to the destination
"""
def plotev(ev, destination):
    
    (k,) = ev.shape 
    x = np.arange(1,k+1,1)
    plt.axis([0,k+1,min(ev),max(ev)])
    plt.plot(x, ev, 'ro')
    plt.xlabel('# of features', fontsize=12)
    plt.ylabel('Eigen Values', fontsize=12)
    plt.title('Plot for finding optimal # of features for PCA')
    plt.grid(True)
    plt.savefig(destination)
    plt.close()

"""
Input: Matrix X to be projected
Array u of means
Matrix Z od learned information
Output: Matrix X_proj, i.e, projected X matrix
"""
def pcaproj(X, u, Z):
    
    # Get # of samples
    n = X.shape[0]
    # Helper matrix  
    ones = np.ones([n,1]) 
    # Center the data
    X_cent = X - ones*u
    # Project the data
    X_proj = np.dot(X_cent,Z)
    return X_proj
    

"""
Scatter plot x, y group by label and save plot at destination (PCA by hand)
"""
def plotpca(x, y, label, destination):
    
    df = pd.DataFrame(dict(x=x, y=y, label=label))
    groups = df.groupby('label')    
    uniq = len(list(set(label)))
    # Set the color map to match the number of labels
    hot = plt.get_cmap('hot')
    cNorm  = colors.Normalize(vmin=0, vmax=uniq)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                color=scalarMap.to_rgba(name), label=name)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)    
    
    plt.xlabel('1st PCA Component', fontsize=12)
    plt.ylabel('2nd PCA Component', fontsize=12)
    plt.title('Data plot in 2D using PCA (w/o Library)')
    plt.savefig(destination)
    plt.close()
    
"""
Plots two plots, first to know optimum value of # of features in rpojected data,
second to plot projected data against different classes
Output:
Optimum # of features
"""
def getplotsPCA(X, Y):
    
    X_train = X
    p = 0.95
    u, Z, ev, F = pcalearn(X_train, p)
    destination = '../Data/Plots/Visualize/plotEigenValues.pdf'
    plotev(ev, destination)
    
    X_test = X
    X_proj = pcaproj(X_test, u, Z)
    
    x_2, y_2 = getVisData(X_proj, Y)
    # Select 1st PCA Component
    x_p = x_2[:,0]
    # Select 2nd PCA Component
    y_p = x_2[:,1]
    # Actual class
    labels = y_2
    destination = '../Data/Plots/Visualize/plotPCAManual.pdf'
    plotpca(x_p, y_p, labels, destination)
    
    return F


"""
Input:
C: penalty for SVM
loss: loss function for SVM
X_train: Training X values
y_train: Training y values
X_test: Testing X values
y_test: Testing y values
Output:
f1_score: F Score of the resulting classifier
"""
def getF1Score(C, loss, X_train, y_train, X_test, y_test):
    
    clf = svm.LinearSVC(penalty='l2', loss=loss, dual=True, tol=0.0001, 
                        C=C, multi_class='ovr', fit_intercept=True, 
                        intercept_scaling=1, class_weight=None, verbose=0, 
                        random_state=None, max_iter=1000)
                        
    #Train Linear SVM Model
    clf.fit(X_train, y_train)    
    #Predict labels for cross validation set
    y_pred = clf.predict(X_test)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for j in range(y_test.shape[0]):
        if y_test[j]==1 and y_pred[j]==1:
            tp+=1
        if y_test[j]==-1 and y_pred[j]==1:
            fp+=1
        if y_test[j]==1 and y_pred[j]==-1:
            fn+=1
        if y_test[j]==-1 and y_pred[j]==-1:
            tn+=1
    recall = tp/(tp+fn)
    if tp==0 and fp==0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if recall==0 and precision==0:
        f1_score = 0
    else:
        f1_score = 2*recall*precision/(recall+precision)    
    return f1_score
    
    
"""
Input:
C: penalty for SVM
loss: loss for SVM
X: Training Matrix (M x N)
Y: Labels (M x #classes)
k: # of folds
c: class # (1 to 10)
Output:
f: Array of optimum # of features
z: Array of F1 Score for different folds
mu: Mean of F1 Scores
var: Var of F1 Scores
"""
def kfoldcv(C, loss, X, Y, k, c):
    
    n = X.shape[0]
    z = np.zeros(k)
    f = np.zeros(k)
    
    for i in range(k):
        #print("Fold # ",i+1)
        # Create temporary array to store the index of ith fold
        a = np.zeros(n)
        a[int(np.floor(n*i/k)):int(np.floor((i+1)*n/k))] = 1
        # Get training set
        X_train = X[a==0]
        y_train = Y[a==0,int(c-1)]
        # Get testing set
        X_test = X[a==1]
        y_test = Y[a==1,int(c-1)]
        
        p = 0.95
        u, Z, ev, F = pcalearn(X_train, p)
        X_train_proj = pcaproj(X_train, u, Z)
        X_test_proj = pcaproj(X_test, u, Z)
        
        z[i] = getF1Score(C, loss, X_train_proj, y_train, X_test_proj, y_test)   
        f[i] = F
        
    mu = np.mean(z)
    var = np.var(z)    
    
    return f, z, mu, var
    

"""
Input: 
z: Array with mean of different folds
C: penalty of SVM
loss: loss of SVM
destination: path to save plot to
Output:
Plot of F1 Score for different folds
"""
def plotF1Score(z, C, loss, destination):
    
    (k,) = z.shape 
    x = np.arange(1,k+1,1)
    plt.axis([0,k+1,min(z)-0.02,max(z)+0.02])
    plt.plot(x, z, 'ro')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    title = 'F1 Score for different folds (C = ' + str(C) + ', loss = ' + loss + ')' 
    plt.title(title)
    plt.grid(True)
    plt.savefig(destination)
    plt.close()
    
"""
Input:
X: Input Matrix (M x N)
Y: Label Matrix (M x 10)
C: array of penalty
loss: array of loss
k: number of folds
c: class number (1 to 10)
Output:
f: array of optimum # of features in each fold
z: matrix of F1 score for each experiment
mu: array of mean F1 score for each experiment
var: array of variance of F1 score for each experiment
"""
def tuneModel(X, Y, C, loss, k, c):

    i = 0
    itera = len(C)*len(loss)
    z = np.zeros([itera,k])
    mu = np.zeros(itera)
    var = np.zeros(itera)
    
    for losd in loss:          
        for cd in C:
            expno = i+1
            print "Experiment No. ", expno
            print "Training for loss: ",losd, "and C: ",cd
            f, z[i,:], mu[i], var[i] = kfoldcv(cd, losd, X, Y, k, c)
            destination = '../Data/Plots/SVM/plotF1Exp' + str(expno) + '.pdf'
            plotF1Score(z[i,:], cd, losd, destination)
            i = i+1
    
    return f, z, mu, var


"""
Input: 
f: Array with # of features of different folds
destination: path to save plot to
Output:
Plot of # of features for different folds
"""
def plotNumFeat(f, destination):
    
    (k,) = f.shape 
    x = np.arange(1,k+1,1)
    plt.axis([0,k+1,min(f)-10,max(f)+10])
    plt.plot(x, f, 'ro')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Optimum # of features', fontsize=12)
    title = '# of features for different folds' 
    plt.title(title)
    plt.grid(True)
    plt.savefig(destination)
    plt.close()

"""
Input:
z: Matrix having F1 scores for each experiment for each fold (#exp x #folds)
mu: Array of average F1 score for each experiment (#exp,)
destination: location to save plot to
Output:
Plots average F1 score for different experiments, first with all folds,
second with all folds except last
mu2: array of mean of F1 score for each experiment for all folds except last
var2: array of variance of F1 score for each experiment for all folds except last
"""
def plotAvgF1(z, mu, destination):
    
    (exps,k) = z.shape
    w = z[:,:(k-1)]
    mu2 = np.zeros(exps)
    var2 = np.zeros(exps)
    for i in range(exps):
        mu2[i] = np.mean(w[i,:])
        var2[i] = np.var(w[i,:])
        
    x = np.arange(1,exps+1,1)
    plt.axis([0,exps+1,min(mu)-0.01,max(mu2)+0.01])
    plot1, = plt.plot(x, mu, 'bo', label = 'All folds')
    plot2, = plt.plot(x, mu2, 'ro', label = 'but last')
    plt.legend(handles=[plot1, plot2], loc=4)
    plt.xlabel('Experiment #', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.title('Average F1 Score for different experiments')
    plt.grid(True)
    plt.savefig(destination)
    plt.close()

    return mu2, var2