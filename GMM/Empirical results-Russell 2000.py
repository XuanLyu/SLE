#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import testjx
from scipy.optimize import minimize
from sympy.solvers import solve
from sympy import Symbol
from scipy import stats
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings("ignore")


# In[4]:


C_r=pd.read_excel('1999-2019consumption.xlsx')


# In[5]:


import pandas as pd
import openpyxl
Income=34*4.28*pd.read_excel('mothlyIncome_average.xlsx')#increasing rate


# In[6]:


BMI_1=pd.read_csv('AAPL.csv')
BMI_2=pd.read_csv('INTC.csv')
BMI_3=pd.read_csv('^DJI.csv')
BMI_3=pd.read_csv('IBM.csv')
BMI_4=pd.read_csv('MSFT.csv')
BMI_=(BMI_1+BMI_2+BMI_4+BMI_3)
BMI=pd.read_csv('^RUT.csv')
BMI_R=pd.read_csv('^GSPC.csv')
lose1=BMI['Close']
lose2=BMI_['Close']/4
lose3=BMI_R['Close']

loglose1=np.log(lose1)
loglose2=np.log(lose2)
loglose3=np.log(lose3)
r_d1=loglose1.diff(1)[1:]
r_d2=loglose2.diff(1)[1:]
r_d3=loglose3.diff(1)[1:]
BMI['real return']=np.hstack((np.array([0]),np.array(lose1[1:])/np.array(lose1[:-1])))
BMI['consumption_r']=C_r/100+1



# In[13]:


BMI=pd.read_csv('^RUT.csv')
lose1=BMI['Close']
loglose1=np.log(lose1)
r_d1=loglose1.diff(1)[1:]

BMI['real return']=np.hstack((np.array([0]),np.array(lose1[1:])/np.array(lose1[:-1])))
BMI['consumption_r']=C_r/100+1


# In[14]:


C=[0.89*Income['average_income'][0]]
for i in range(251):
    C.append(C[-1]*BMI['consumption_r'][i+1])
BMI['consumption']=C
BMI['Income']=list(Income['average_income'])


# # GMM estimation of classical model
# 
# 

# In[18]:


def moment_BMI(x):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*BMI['real return'][2:]*BMI['consumption_r'][2:]**x1-1)
    m2=m1*np.array(BMI['consumption_r'][1:-1])
    m3=m1*np.array(BMI['real return'][1:-1])
    return np.array([np.mean(m1),np.mean(m2),np.mean(m3)])
def weight_BMI(x):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*BMI['real return'][2:]*BMI['consumption_r'][2:]**x1-1)
    m2=m1*np.array(BMI['consumption_r'][1:-1])
    m3=m1*np.array(BMI['real return'][1:-1])
    V=np.vstack((np.array(m1),np.array(m2),np.array(m3)))
    return V.dot(V.T)/len(m1)
def J_BMI(x):
   
    m=moment_BMI(x)
    return m.dot(m)
res=minimize(J_BMI,[1,-1])
def J_w_BMI(x):
    cov=np.linalg.inv(weight_BMI(res.x))
    m=moment_BMI(x).reshape(3,1)
    return (m.T.dot(cov)).dot(moment_BMI(x))
def J_w_BMI_(x):
    cov=np.linalg.inv(weight_BMI(res_w.x))
    m=moment_BMI(x).reshape(3,1)
    return (m.T.dot(cov)).dot(moment_BMI(x))

res_w=minimize(J_w_BMI_,[1,-1])

res_w.x,res_w.fun*252,stats.chi2.cdf(res_w.fun*252,1) 


# In[19]:


def Dmoment(x):
    x0=float(x[0])
    x1=float(x[1])
    d1=np.vstack((np.array((BMI['real return'][2:]*BMI['consumption_r'][2:]**x1)/250),np.array((x0*np.log(BMI['consumption_r'][2:])*BMI['real return'][2:]*BMI['consumption_r'][2:]**x1)/250)))
    d2=np.vstack((np.ones(250),np.array(BMI['real return'][1:-1]),np.array(BMI['consumption_r'][1:-1]))).T
    return d1.dot(d2)
def Asyvar(x):
    sigma=np.linalg.inv(Dmoment(x).dot(np.linalg.inv(weight_BMI(x))).dot(Dmoment(x).T))
    return np.sqrt(np.diag(sigma)/252)
Asyvar(res_w.x)


# # GMM estimation of max-min  model

# In[20]:


def group(n):
    r=pd.DataFrame([[0]*(len(r_d1)+1-n) for j in range(n)])
    for i in range(len(r_d1)+1-n):
        r[i]=r_d1.tolist()[i:i+n]
    groupmean=np.mean(r)
    return list(groupmean[groupmean==max(groupmean)].index)
G=[]
for k in range(125,252):
    G+=group(k)##均值最大所在组的时间指标
PG=pd.DataFrame({'length':np.array(range(125,252)),'start':G,'end':G+np.array(range(125,252))})##时间区间
# PG['end']=np.array(G['start'])+np.array(G['length'])
def group_(n):
    r=pd.DataFrame([[0]*(len(r_d1)+1-n) for j in range(n)])
    for i in range(len(r_d1)+1-n):
        r[i]=r_d1.tolist()[i:i+n]
    groupmean=np.mean(r)
    return list(groupmean[groupmean==min(groupmean)].index)
G_=[]##均值最小所在组的时间指标
for k in range(125,252):
    G_+=group_(k)
PG_=pd.DataFrame({'length':np.array(range(125,252)),'start':G_,'end':G_+np.array(range(125,252))})
def BMI_moment_(x,n1,n2):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*BMI['real return'][n1+2:n2+1]*BMI['consumption_r'][n1+2:n2+1]**x1-1)
    m2=m1*np.array(BMI['consumption_r'][n1+1:n2])
    m3=m1*np.array(BMI['real return'][n1+1:n2])
    return np.array([np.mean(m1),np.mean(m2),np.mean(m3)])
def BMI_weight_(x,n1,n2):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*BMI['real return'][n1+2:n2+1]*BMI['consumption_r'][n1+2:n2+1]**x1-1)
    m2=m1*np.array(BMI['consumption_r'][n1+1:n2])
    m3=m1*np.array(BMI['real return'][n1+1:n2])
    V=np.vstack((np.array(m1),np.array(m2),np.array(m3)))
    return V.dot(V.T)/len(m1)
def BMI_J_(x,n1,n2):
    m=BMI_moment_(x,n1,n2)
    return m.dot(m)
# res=minimize(lambda x:BMI_J_(x,,245),[1,-1])
def BMI_JJ_w(x,n1,n2):
    w=np.linalg.inv(BMI_weight_(res.x,n1,n2))
    m=BMI_moment_(x,n1,n2).reshape(3,1)
    return (m.T.dot(w)).dot(BMI_moment_(x,n1,n2))


# In[21]:


def dmoment(x,n1,n2):
    x0=float(x[0])
    x1=float(x[1])
    d1=np.vstack((np.array((BMI['real return'][n1+2:n2+1]*BMI['consumption_r'][n1+2:n2+1]**x1)/(n2-n1-1)),np.array((x0*np.log(BMI['consumption_r'][n1+2:n2+1])*BMI['real return'][n1+2:n2+1]*BMI['consumption_r'][n1+2:n2+1]**x1)/(n2-n1-1))))
    d2=np.vstack((np.ones(n2-n1-1),np.array(BMI['real return'][n1+1:n2]),np.array(BMI['consumption_r'][n1+1:n2]))).T
    return d1.dot(d2)
def asyvar(x,n1,n2):
    sigma=np.linalg.inv(dmoment(x,n1,n2).dot(np.linalg.inv(BMI_weight_(x,n1,n2))).dot(dmoment(x,n1,n2).T))
    return np.sqrt(np.diag(sigma)/(n2-n1))


# In[22]:


# Estimations and reject prob for max-min model
para=[]
para_=[]
J_test=[]
var=[]
for n1,n2 in np.array(PG_[['start','end']]):
    res=minimize(lambda x:BMI_J_(x,n1,n2),[1,-1])
    res_w=minimize(lambda x:BMI_JJ_w(x,n1,n2),[1,-1])
    para.append(res.x)
    para_.append(res_w.x)
    J_test.append(res_w.fun*(n2-n1+1))
    var.append(asyvar(res_w.x,n1,n2))
result=pd.DataFrame(np.array(para_),columns=['beta','alpha'])
result['J-test']=J_test
result['start']=PG_['start']
result['end']=PG_['end']
result['window']=PG_['end']-PG_['start']


result['prob']=[stats.chi2.cdf(result['J-test'][k],1) for k in range(len(result['J-test']))]


# In[23]:


result['asyvar-beta']=np.array(var)[:,0]
result['asyvar-alpha']=np.array(var)[:,1]


# In[24]:


result[result['alpha']<0][['beta','alpha','J-test','window','prob','asyvar-beta','asyvar-alpha']]


# In[25]:


result.to_excel('max-min model for ^RUT.xls')


# # GMM estimation of max-max  model

# In[26]:


# Estimations and reject prob for max-max model
para=[]
para_=[]
J_test=[]
var=[]
for n1,n2 in np.array(PG[['start','end']]):
    res=minimize(lambda x:BMI_J_(x,n1,n2),[1,-1])
    res_w=minimize(lambda x:BMI_JJ_w(x,n1,n2),[1,-1])
    para.append(res.x)
    para_.append(res_w.x)
    J_test.append(res_w.fun*(n2-n1+1))
    var.append(asyvar(res_w.x,n1,n2))
result_max=pd.DataFrame(np.array(para_),columns=['beta','alpha'])
result_max['J-test']=J_test
result_max['start']=PG['start']
result_max['end']=PG['end']
result_max['window']=PG['end']-PG['start']


result_max['prob']=[stats.chi2.cdf(result_max['J-test'][k],1) for k in range(len(result_max['J-test']))]
result_max['asyvar-beta']=np.array(var)[:,0]
result_max['asyvar-alpha']=np.array(var)[:,1]


# In[28]:


result_max[result_max['alpha']<0].to_excel('max-max model for ^RUT.xls')


# In[ ]:




