#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import scipy.stats as st

from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import fbeta_score,f1_score,balanced_accuracy_score,recall_score,precision_score
import warnings
warnings.filterwarnings('ignore')
from cvxopt import matrix
from cvxopt import solvers
class Mean_uncertain:
    def __init__(self,model=LogisticRegression()):
#         self.n=n
        self.model=model
        
        
        
    def sigmoid_(self,x,u):
        return 1.0/(1.0+np.exp(-x-u))
    def meanuncertainty(self,x,n):
        r=[]
        for i in range(0,len(x)+1-n,n//10):
            r.append(np.mean(x[i:i+n]))
        return min(r),max(r)
    def error(self,X,Y):
        self.model.fit(X,Y)
        pre_in=(np.dot(X,self.model.coef_.T)+self.model.intercept_).reshape(X.shape[0],)
        return pre_in
    def equa_m(self,x,X,Y,n):
        
        train_err=self.error(X,Y)

        ini_err=Y-self.sigmoid_(train_err,x)##ini_err is the predicted err of training set, based on LR

        for k in range(ini_err.shape[0]):
            if Y[k]==1:
                ini_err[k]=0.5*(Y.shape[0]/sum(Y))*ini_err[k]

        return self.meanuncertainty(ini_err,n)[1]

   
    
    def upper_mean(self,X,Y,n1,n2):
        mean_lr=np.array([fsolve(lambda x:self.equa_m(x,X,Y,n),0.5 ) for n in list(range(10,int(Y.shape[0]/2),n1))+list(range(int(Y.shape[0]/2),Y.shape[0]+1,n2))])
        return mean_lr
    def predict_prob(self,model,x_te,mean_u):
#         self.model.fit(X,Y)
        prob_y=self.sigmoid_((np.dot(x_te,model.coef_.T)+model.intercept_).reshape(x_te.shape[0],),mean_u
             )
        y_train_proba=np.vstack((1-prob_y,prob_y)).T
        
        return y_train_proba
    def predict(self,model,x_te,mean_u):
        prob_y=self.sigmoid_((np.dot(x_te,model.coef_.T)+model.intercept_).reshape(x_te.shape[0],),mean_u
             )

        prdict_y=np.round(prob_y)
        return prdict_y


# In[ ]:




