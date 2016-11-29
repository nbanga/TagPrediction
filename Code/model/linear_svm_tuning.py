import pandas as p
import numpy as np
from sklearn import svm
import time

filename = "../Data/x_vector/X.csv"
X_read = p.read_csv(filename)

filename = "../Data/x_vector/Y.csv"
y_read = p.read_csv(filename)

X_data = np.array(X_read)
[n,d] = X_data.shape

y_data = np.array(y_read)
[n,l] = y_data.shape

n_classes = l
# number of classes you want to train for
# put n_classes = 5 or 10 since there l is 100, and it will long time to run

split = 500
# Take first 3000 data points for training and rest for cross-validation

C = [0.1,0.5,1,10,50,100,1000]
loss = ['hinge','squared_hinge']
penalty = ['l2']

# Get training data
X_train = X_data[:split,:]

# Get cross validation data
X_test = X_data[split:,:]


# function to train the classifiers
# output: theta of size 10 x 4444 (one row per class)
def train_model():
    
    # will store tuned theta vector for the classifiers
    theta = np.array([[0.0]*d]*n_classes)    
    
    for i in range(n_classes):
        
        start = time.time()
        
        y_train = y_data[:split,i]   
        y_test = y_data[split:,i]
        max_f_score = -1
        tuned_C = -1
        tuned_loss = ''
        tuned_penalty = ''
        
        print("Tuning Classifier for Tag #",i+1)
    
        for c in C:
            
            for los in loss:
                
                for pen in penalty:
            
                    """
                    print("C:",c)
                    print("loss:",los)
                    print("penalty:",pen)
                    """
                    
                    clf = svm.LinearSVC(penalty=pen, loss=los, dual=True, 
                                     tol=0.0001, C=c, multi_class='ovr', 
                                     fit_intercept=True, intercept_scaling=1, 
                                     class_weight=None, verbose=0, random_state=None, 
                                     max_iter=1000)
                    
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
                    #accuracy = (tp+tn)/(tp+tn+fp+fn)
                    #error = (fp+fn)/(tp+tn+fp+fn)
                    recall = float(tp)/(tp+fn)
                    if tp==0 and fp==0:
                        precision = 0
                    else:
                        precision = float(tp)/(tp+fp)
                    #specificity = tn/(tn+fp)
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
                        f1_score = 2*recall*precision/(recall+precision)
                    #print("F1 Score:",f1_score)
                    
                    if(f1_score>max_f_score):
                        max_f_score = f1_score
                        tuned_C = c
                        tuned_loss = los
                        tuned_penalty = pen
                        theta[i] = clf.coef_
        
        print("Elapsed Time",time.time()-start)       
        print("Tuned C:",tuned_C)
        print("Tuned loss:",tuned_loss)
        print("Tuned Penalty:",tuned_penalty)
        print("Tuned F1 Score",max_f_score)
        
    return theta
    
#Call train_model to get tuned theta vector
theta = train_model()
