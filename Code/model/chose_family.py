import pandas as p
import numpy as np
from sklearn import svm
import time

# filename = "../Data/x_vector/X.csv"
# X_read = p.read_csv(filename)
#
# filename = "../Data/x_vector/Y.csv"
# y_read = p.read_csv(filename)

#comment above for running skewed
#uncomment below to run for skewed
filename = "../Data/x_vector/X_skewed.csv"
X_read = p.read_csv(filename)
filename = "../Data/x_vector/Y_skewed.csv"
y_read = p.read_csv(filename)

X_data = np.array(X_read)
[n,d] = X_data.shape

y_data = np.array(y_read)
[n,l] = y_data.shape

n_classes = 1
# number of classes you want to train for to pick the family

split = 3000
# Take first 3000 data points for training and rest for cross-validation

#C = [0.1,1,10]
# uncomment above for full run
# comment below line for full run
C = [0.1]
kernel = ['linear','rbf','poly']

# Get training data
X_train = X_data[:split,:]

# Get cross validation data
X_test = X_data[split:,:]

for i in range(n_classes):
    
    start = time.time()

    y_train = y_data[:split,i]   
    y_test = y_data[split:,i]    
    max_f_score = -1
    tuned_C = -1
    tuned_kernel = ''
    
    print "Training Classifier for Tag #",i+1

    for c in C:
        
        for ker in kernel:
            
            print "C:",c
            print "kernel:",ker
	
            clf = svm.SVC(C=c, kernel=ker, degree=3, gamma='auto', coef0=0.0, 
                          shrinking=True, probability=True, tol=0.001, 
                          cache_size=200, class_weight=None, verbose=False, 
                          max_iter=-1, decision_function_shape=None, 
                          random_state=None)
              
            #Train Linear SVM Model
            clf.fit(X_train, y_train) 	

            #Predict labels for cross validation set
            y_pred = clf.predict(X_test)

            tp = 0
            fp = 0
            fn = 0
            tn = 0
            
            for j in range(n-split):
                if y_test[j]==1 and y_pred[j]==1:
                    tp+=1
                if y_test[j]==-1 and y_pred[j]==1:
                    fp+=1
                if y_test[j]==1 and y_pred[j]==-1:
                    fn+=1
                if y_test[j]==-1 and y_pred[j]==-1:
                    tn+=1
                        
            """       
            print("True Postives:",tp)
            print("False Postives:",fp)
            print("False Negatives:",fn)
            print("True Negatives:",tn)
            """
            accuracy = 1.0*(tp+tn)/(tp+tn+fp+fn)
            error = 1.0*(fp+fn)/(tp+tn+fp+fn)
            recall = 1.0*tp/(tp+fn)
            if tp==0 and fp==0:
                precision = 0
            else:
                precision = 1.0*tp/(tp+fp)
            specificity = 1.0*tn/(tn+fp)
            """
            print("Train Accuracy:",clf.score(X_train,y_train))
            print("Test Accuracy:",accuracy)
            print("Test Error:",error)
            print("Recall:",recall)
            print("Precision:",precision)
            print("Specificity:",specificity)
            """
            if recall==0 and precision==0:
                f1_score = 0
            else:
                f1_score = 2.0*recall*precision/(recall+precision)
            print "F1 Score:",f1_score
            print
                
            if(f1_score>max_f_score):
                max_f_score = f1_score
                tuned_C = c
                tuned_kernel = ker
    
    print "Tuned C:\t",tuned_C
    print "Tuned Kernel:\t",tuned_kernel
    print "Tuned F1 Score:\t",max_f_score