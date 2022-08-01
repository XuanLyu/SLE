#!/usr/bin/env python
# coding: utf-8

# In[2]:


from SLE_algorithm import method
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import scipy.stats as st

from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import fbeta_score,f1_score,balanced_accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

import warnings
warnings.filterwarnings('ignore')

class E_Teaser:
    def __init__(self,E=True,model=method.Mean_uncertain(),fix=10,master_mu=0,slave_mu=0):
        
#         self.n=n
        self.model=model
        self.fix=fix
        self.m=master_mu #m=0,m_=0时对应普通Teaser
        self.m_=slave_mu
        self.E=E #True:使用两个master来增强边界
    def pre_models(self,train_x,train_y):

        slaver_xin=[]
        master_xin=[]


        for i in range(self.fix,train_x.shape[1]-self.fix+1,self.fix):
            lr=LogisticRegression()
            lr.fit(train_x[:,:i],train_y)
            slaver_xin.append(lr)
            

            y_train_label=self.model.predict(lr,train_x[:,:i],self.m_)
            y_train_proba=self.model.predict_prob(lr,train_x[:,:i],self.m_)

            s_feature=np.vstack((y_train_label,y_train_proba[:,0],y_train_proba[:,1],
                        np.max(y_train_proba,axis=1)-np.min(y_train_proba,axis=1))).T
            if self.E:
                index_0=np.argwhere(train_y==0).reshape(-1)
                index_1=np.argwhere(train_y==1).reshape(-1)
                s_feature_1=s_feature[index_1]
                s_feature_0=s_feature[index_0]
                if sum(np.equal(np.argmax(y_train_proba[index_1],axis=1
                 ),train_y[index_1])):

                    clf_1 = OneClassSVM(gamma='auto').fit(s_feature_1[np.equal(np.argmax(y_train_proba[index_1],axis=1
                 ),train_y[index_1])])
                else: clf_1 = OneClassSVM(gamma='auto').fit(s_feature[np.equal(np.argmax(y_train_proba,axis=1
                     ),train_y)])
                if sum(np.equal(np.argmax(y_train_proba[index_0],axis=1
             ),train_y[index_0])):
                    clf_0 = OneClassSVM(gamma='auto').fit(s_feature_0[np.equal(np.argmax(y_train_proba[index_0],axis=1
                 ),train_y[index_0])])

                else:clf_0 = OneClassSVM(gamma='auto').fit(s_feature[np.equal(np.argmax(y_train_proba,axis=1
                     ),train_y)])




                master_xin.append([clf_0,clf_1])
            else:
                clf = OneClassSVM(gamma='auto').fit(s_feature[np.equal(np.argmax(y_train_proba,axis=1
                     ),train_y)])
                master_xin.append(clf)
        return slaver_xin,master_xin
    def x_test_lr(self,index,train_x,train_y,x,slaver_xin,master_xin):
        
        
        v=1
        label_con=[100]
        for i in range(self.fix,x.shape[1]+1-self.fix,self.fix):
            
            y_test_proba= self.model.predict_prob(slaver_xin[int(i/self.fix-1)],x[index][:i].reshape(1,-1),self.m).reshape(-1)
            y_test_label=np.argmax(y_test_proba)
            s_feature_te=np.array([y_test_label,y_test_proba[0],y_test_proba[1],
                            max(y_test_proba)-min(y_test_proba)]).T
    
            if self.E:
                y_te_index=max(master_xin[int(i/self.fix-1)][0].predict(s_feature_te.reshape(1,-1)),master_xin[int(i/self.fix-1)][1].predict(s_feature_te.reshape(1,-1)))
            else: y_te_index=master_xin[int(i/self.fix-1)].predict(s_feature_te.reshape(1,-1))
                
            if y_te_index==1:
                label_con.append(int(y_test_label))
                if label_con[-1]==label_con[-2]:
                    v+=1
            if v==2:
                break
        if label_con[-1]==100:
            lr=LogisticRegression()
            lr.fit(train_x,train_y)

    #         y_test_proba=model.predict(lr,)sigmoid_((np.dot(x[index].reshape(1,-1),lr.coef_.T)+lr.intercept_),
    #       k).reshape(-1)
            L=self.model.predict(lr,x[index].reshape(1,-1),self.m)
            label_con.append(int(L))
            decision_t=1
        else:
            decision_t=i/x.shape[1]
        y_final=label_con[-1]
        return decision_t,y_final
    def predict(self,train_x,train_y,test_x):
       
        slaver_xin,master_xin=self.pre_models(train_x,train_y)
       
        xlr_T_star=[]
        xlr_Label=[]
        for j in range(test_x.shape[0]):
            t_star,label=self.x_test_lr(j,train_x,train_y,test_x,slaver_xin,master_xin)
            xlr_T_star.append(t_star)
            xlr_Label.append(label)
        return xlr_T_star, xlr_Label


# In[ ]:




