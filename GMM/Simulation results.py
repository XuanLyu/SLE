#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import testjx
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# In[3]:


P_0=10
W_0=200
alpha=-0.8
beta=0.98


# In[4]:


simu_p_1=pd.read_excel('simulated_p.xls')
fig, ax = plt.subplots()
simu_p_1[0].plot()
plt.xlabel('Time index: t')
plt.ylabel(' Price: $S_t$')
np.random.seed(21)
ep=np.random.normal(0,0.09,252)


# In[5]:


simu_p=[10]+list(simu_p_1[0])
m_group_=np.log(np.array(simu_p[1:])/np.array(simu_p[:-1]))-ep


# In[6]:


min(m_group_),max(m_group_)


# In[7]:


np.random.seed(321)
e=np.random.normal(0,0.0001,252)
C_0=1
C_=[C_0]

for i in range(252):

    C_.append(((C_[-1]**alpha-e[i])/(beta*np.exp(ep+max(m_group_))[i]))**(1/alpha))


# In[8]:


pd_data_=pd.DataFrame(C_,columns=['consumption'])
pd_data_['consumption_r']=pd_data_['consumption'][1:]/pd_data_['consumption'][:-1]
pd_data_['consumption_r'][1:]=np.array(C_[1:])/np.array(C_[:-1])
pd_data_['real return']=np.hstack((np.array([0]),np.exp(ep+m_group_)))
pd_data_.describe()


# In[9]:


m_u1=[]
m_u2=[]
for k in range(10,253):
    u1,u2=testjx.testjx.meanuncertainty(ep+m_group_,k)
    m_u1.append(u1)
    m_u2.append(u2)


# #  GMM estimation of classical model

# In[10]:


def moment(x):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*pd_data_['real return'][2:]*pd_data_['consumption_r'][2:]**x1-1)
    m2=m1*np.array(pd_data_['consumption_r'][1:-1])
    m3=m1*np.array(pd_data_['real return'][1:-1])
    return np.array([np.mean(m1),np.mean(m2),np.mean(m3)])
def weight(x):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*pd_data_['real return'][2:]*pd_data_['consumption_r'][2:]**x1-1)
    m2=m1*np.array(pd_data_['consumption_r'][1:-1])
    m3=m1*np.array(pd_data_['real return'][1:-1])
    V=np.vstack((np.array(m1),np.array(m2),np.array(m3)))
    return V.dot(V.T)/len(m1)
def J(x):
    m=moment(x)
    return m.dot(m)
res=minimize(J,[1,-1])
def J_w(x):
    w=np.linalg.inv(weight(res.x))
    m=moment(x).reshape(3,1)
    return (m.T.dot(w)).dot(moment(x))
res_w=minimize(J_w,[1,-1])


# In[11]:


#Classical GMM estimation and the corresponding reject Prob
res.x,res_w.x,res_w.fun*252,stats.chi2.cdf(res_w.fun*252,1)


# # GMM of Max-max model

# In[12]:


def group_s(n):
    r=pd.DataFrame([[0]*(len(ep+m_group_)+1-n) for j in range(n)])
    for i in range(len(ep+m_group_)+1-n):
        r[i]=(ep+m_group_).tolist()[i:i+n]
    groupmean=np.mean(r)
    return list(groupmean[groupmean==max(groupmean)].index)
G=[]
for k in range(10,253):
    G+=group_s(k)
PG=pd.DataFrame({'length':np.array(range(10,253)),'start':G,'end':G+np.array(range(10,253))})
# PG['end']=np.array(G['start'])+np.array(G['length'])

def BMI_moment_s(x,n1,n2):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*pd_data_['real return'][n1+2:n2+1]*pd_data_['consumption_r'][n1+2:n2+1]**x1-1)
    m2=m1*np.array(pd_data_['consumption_r'][n1+1:n2])
    m3=m1*np.array(pd_data_['real return'][n1+1:n2])
    return np.array([np.mean(m1),np.mean(m2),np.mean(m3)])
def BMI_weight_s(x,n1,n2):
    x0=float(x[0])
    x1=float(x[1])
    m1=(x0*pd_data_['real return'][n1+2:n2+1]*pd_data_['consumption_r'][n1+2:n2+1]**x1-1)
    m2=m1*np.array(pd_data_['consumption_r'][n1+1:n2])
    m3=m1*np.array(pd_data_['real return'][n1+1:n2])
    V=np.vstack((np.array(m1),np.array(m2),np.array(m3)))
    return V.dot(V.T)/len(m1)
def BMI_J_(x,n1,n2):
    m=BMI_moment_s(x,n1,n2)
    return m.dot(m)
# res=minimize(lambda x:BMI_J_(x,,245),[1,-1])
def BMI_JJ_w(x,n1,n2):
    w=np.linalg.inv(BMI_weight_s(res.x,n1,n2))
    m=BMI_moment_s(x,n1,n2).reshape(3,1)
    return (m.T.dot(w)).dot(BMI_moment_s(x,n1,n2))
para=[]
para_=[]
J_test=[]
for n1,n2 in np.array(PG[['start','end']]):
    res=minimize(lambda x:BMI_J_(x,n1,n2),[1,-1])
    res_w=minimize(lambda x:BMI_JJ_w(x,n1,n2),[1,-1])
    para.append(res.x)
    para_.append(res_w.x)
    J_test.append(res_w.fun*(n2-n1))
result=pd.DataFrame(np.array(para_),columns=['beta','alpha'])
result['J-test']=J_test
result['start']=PG['start']
result['ebd']=PG['end']
result['window']=PG['end']-PG['start']
result['prob']=[stats.chi2.cdf(result['J-test'][k],1) for k in range(len(result['J-test']))]
result['suppermean']=m_u2


# In[13]:


simu_1=result[result['beta']<1]
simu_1=simu_1[simu_1['alpha']<0]
fig, ax = plt.subplots()
simu_1['prob'][1:].plot()
plt.vlines(35,0,1,'y','--')
plt.xlabel('Window size: N')
plt.ylabel('Reject Prob: $\hat{P}_N$ ')


# In[14]:


simu_2=simu_1[simu_1['prob']<0.8]
fig, ax = plt.subplots()
rect_1=ax.plot(simu_2['window'][1:16],simu_2['suppermean'][1:16],label='')
ax2 = ax.twinx()
rect_2=ax2.scatter(simu_2['window'][1:16],simu_2['prob'][1:16],s=15,c='y')
ax.legend((rect_1[0], rect_2), ('$\hat{\overline{\mu}}_N$', '$\hat{P}_N$'))
ax.set_xlabel('Window size: N')
ax.set_ylabel('Estimation of Upper Mean: $\hat{\overline{\mu}}_N$')
ax2.set_ylabel('Reject Prob: $\hat{P}_N$')


# In[15]:


result.to_excel('re_simulated result_maxmax.xls')


# # GMM of Max-min model

# In[16]:


def group_(n):
    r=pd.DataFrame([[0]*(len(ep+m_group_)+1-n) for j in range(n)])
    for i in range(len(ep+m_group_)+1-n):
        r[i]=(ep+m_group_).tolist()[i:i+n]
    groupmean=np.mean(r)
    return list(groupmean[groupmean==min(groupmean)].index)
G_=[]##均值最小所在组的时间指标
for k in range(10,253):
    G_+=group_(k)
PG_=pd.DataFrame({'length':np.array(range(10,253)),'start':G_,'end':G_+np.array(range(10,253))})


# In[17]:


para=[]
para_=[]
J_test=[]
for n1,n2 in np.array(PG_[['start','end']]):
    res=minimize(lambda x:BMI_J_(x,n1,n2),[1,-1])
    res_w=minimize(lambda x:BMI_JJ_w(x,n1,n2),[1,-1])
    para.append(res.x)
    para_.append(res_w.x)
    J_test.append(res_w.fun*(n2-n1))
result_1=pd.DataFrame(np.array(para_),columns=['beta','alpha'])
result_1['J-test']=J_test
result_1['start']=PG_['start']
result_1['ebd']=PG_['end']
result_1['window']=PG_['end']-PG_['start']
result_1['prob']=[stats.chi2.cdf(result_1['J-test'][k],1) for k in range(len(result_1['J-test']))]
result_1['lowermean']=m_u1


# In[18]:


result_1.to_excel('re_simulated result_maxmin.xls')


# In[ ]:




