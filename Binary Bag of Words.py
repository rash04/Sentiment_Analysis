"""
COMP 551 (Applied Machine Learning) Assignment 3 Question 2
"Sentiment Classification" - Binary Bag of Words
Name: RASHIK HABIB
McGill University
Date: 17th February, 2018
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import metrics
from random import randint
from scipy import sparse
from scipy.sparse import vstack
import copy

print_on = True
test_on = False
dataset = "yelp"

RandMaj_on = True
NB_on = False
DT_on = False
SVM_on = False

if print_on:
    print("*********************" + dataset + "************************")

"""-----------------------------DATA PROCESSING-----------------------------"""
def gen_bag_of_words(file_name):
    
    file_input = open(file_name, 'r')   
    for i,l in enumerate(file_input):
        pass
    lines = i+1
    file_input.close()    
    
    y_true = []
    X_bin_bag = np.zeros((lines ,10000), dtype=int)
    X_freq_bag = np.zeros((lines ,10000), dtype=int)    
    
    review_index = 0    
    
    file_input = open(file_name, 'r')
    
    for review in file_input.readlines():    
        
        (X, y) = review.strip('\n').split('\t')
        y_true.append(int(y))  
        X = X.split(' ')[:-1]
        
        for word_index in X:
            
            word_index = int(word_index)
            X_bin_bag[review_index][word_index] = 1
        
        review_index = review_index + 1
        
    file_input.close()
    return (sparse.csr_matrix(X_bin_bag), y_true)
   
#Get the bin bag of words, freq bag of words and true classification values
file_name =  "hwk3_datasets/" +str(dataset) + "-train-submit.txt"
X_bb_train, y_true_train = gen_bag_of_words(file_name)

file_name =  "hwk3_datasets/" +str(dataset) + "-valid-submit.txt"    
X_bb_valid, y_true_valid = gen_bag_of_words(file_name)
 
file_name =  "hwk3_datasets/" +str(dataset) + "-test-submit.txt"  
X_bb_test, y_true_test = gen_bag_of_words(file_name)  

predef_trainset = [-1]*X_bb_train.shape[0]
predef_validset = [0]*X_bb_valid.shape[0]
predef_dataset = np.concatenate((predef_trainset,predef_validset), axis=0)
ps = PredefinedSplit(predef_dataset)    

#for hyperparameter tuning (gridsearchcv)
X = vstack((X_bb_train, X_bb_valid))     #sparse matrix      
Y = copy.copy(y_true_train)
Y.extend(y_true_valid)
Y = np.array(Y)    
    
"""----------------------RANDOM & MAJORITY CLASSIFIERS----------------------"""
if RandMaj_on:
    
    #random classifier - since y_true is shuffled, we can randomly generate numbers
    #between 1 to 5 in order to form out predicted values (y_random)
    y_random = []
    for i in range(len(y_true_test)):
        y_random.append(randint(1,5))
    F1_random = f1_score(y_true_test, y_random, average='macro')
    
    
    #majority classifier - finds the most frequently occuring class in the training
    #set and classifies all test reviews to be that class
       
    y_train = []
    for review in open("hwk3_datasets/" +str(dataset) + "-train-submit.txt", "r").readlines():
        y_train.append(int(review[-2]))
    y_majority = [np.argmax(np.bincount(y_train))]*len(y_true_test)
    
    F1_majority = f1_score(y_true_test, y_majority, average='macro')
    
    print("*--------Random & Major--------*")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    print("Random Classifier: " + str(F1_random*100) + str('%'))
    print("Majority Classifier: " + str(F1_majority*100) + str('%'))
    print("*------------------------------*")

"""-----------------------------NAIVE BAYES---------------------------------"""
if NB_on:    
    
    NB = BernoulliNB()
    parameters = {'alpha' : np.linspace(0.0, 1.0, 100)}    
    
    clf_NB = GridSearchCV(NB, parameters, cv=ps, scoring=metrics.make_scorer(f1_score, average='macro'))
    clf_NB.fit(X, Y)
    
    results = clf_NB.cv_results_
    alpha =   clf_NB.best_params_['alpha']  
    
    print("*------------NB--------------*")
    print("Best value of alpha: " + str(alpha) )
    print("Validation score: " + str(clf_NB.best_score_ * 100) + str('%'))    
    
    NB.fit(X_bb_train, y_true_train)    
    y_bb_pred_test = NB.predict(X_bb_test)   
    
    F1_bb_NB_test = f1_score(y_true_test, y_bb_pred_test, average='macro')    
    
    print("Resulting Test score: " + str(F1_bb_NB_test*100) + str('%'))
    print("*-----------------------------*")    
    

"""-----------------------------DECISION TREES------------------------------"""
if DT_on:
    
    DT = DecisionTreeClassifier(max_depth=34, min_samples_leaf=35, random_state=0)
    parameters = {'max_depth':np.linspace(33, 35, 10, dtype=int), 'min_samples_leaf':np.linspace(33, 35, 10, dtype=int)}
    
    clf_DT = GridSearchCV(DT, parameters, cv=ps, scoring=metrics.make_scorer(f1_score, average='macro'))    
    clf_DT.fit(X, Y)
    
    results = clf_DT.cv_results_ 
    max_depth = clf_DT.best_params_['max_depth']
    min_samples_leaf = clf_DT.best_params_['min_samples_leaf']    
    
    print("*------------DTs--------------*")
    print("Max depth of tree: " + str(max_depth) )
    print("Min samples for a leaf: " + str(min_samples_leaf) )
    print("Validation score: " + str(clf_DT.best_score_ * 100) + str('%'))        

    DT.fit(X_bb_train, y_true_train)    
    y_bb_pred_test = DT.predict(X_bb_test)   
    
    F1_bb_DT_test = f1_score(y_true_test, y_bb_pred_test, average='macro')
    
    print("Resulting Test score: " + str(F1_bb_DT_test*100) + str('%'))
    print("*-----------------------------*")    
    
"""---------------------------------SVM-------------------------------------"""
if SVM_on:
    
    SVC = LinearSVC()
    parameters = {'C': np.linspace(0.001,0.1, 20) }
    
    clf_SVC = GridSearchCV(SVC, parameters, cv=ps, scoring=metrics.make_scorer(f1_score, average='macro'))    
    clf_SVC.fit(X, Y)    
    
    results = clf_SVC.cv_results_    
    C = clf_SVC.best_params_['C']
    
    print("*------------SVM--------------*")
    print("Best value of C: " + str(C) )
    print("Validation score: " + str(clf_SVC.best_score_ * 100) + str('%'))
    
    SVC.fit(X_bb_train, y_true_train)    
    y_bb_pred_test = SVC.predict(X_bb_test)
    
    F1_bb_SVC_test = f1_score(y_true_test, y_bb_pred_test, average='macro')   

    print("Resulting Test score: " + str(F1_bb_SVC_test*100) + str('%'))
    print("*-----------------------------*")

if test_on:
    print()


if print_on:
    print("*************************************************")
