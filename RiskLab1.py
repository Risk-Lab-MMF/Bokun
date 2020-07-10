# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:48:24 2020

@author: BHuang
"""
import pandas as pd
import numpy as np
from cvxopt import matrix
# from cvxopt.blas import dot 
from cvxopt.solvers import qp, options,lp
from xgboost.sklearn import XGBClassifier
from functools import reduce
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def last_day_of_month(any_day):
    """
    获取获得一个月中的最后一天
    :param any_day: 任意日期
    :return: string
    """
    next_month = any_day.replace(day=28) + timedelta(days=4)  # this will never fail
    return next_month - timedelta(days=next_month.day)


def GeoMean(df):
    # name=df.columns
    [timeN,assetN]=df.shape
    GeoMean=[]
    for i in range(assetN):
        GeoMean.append(np.prod(df.iloc[:,i]+1)**(1/timeN)-1)
    GeoMean=np.array(GeoMean)
    return GeoMean


def sd(w,cov):
    # return np.sqrt(dot(w,Return))
    return float(np.sqrt(reduce(np.dot, [w, cov, w.T])))

def XGBoost(returns,factRet):
    """
    :param return:
    :param facret:
    :param a: lambda
    """
    [timeN,factorN] = factRet.shape
    [timeN,assetN] = returns.shape
    f_bar=[]
    for i in range(factorN):
        f_bar.append(np.prod(factRet.iloc[:,i]+1)**(1/timeN)-1)
        
    colName=list(factRet.columns)
    f_bar=pd.DataFrame(f_bar).T
    f_bar.columns=colName
    xgb = XGBClassifier(learning_rate=0.1,n_estimators=10,
                                max_depth=7,min_child_weight=2,
                                gamma=0.2,subsample=0.8,
                                colsample_bytree=0.6,objective='reg:linear',
                                scale_pos_weight=1,seed=10) 
    mu=[]
    for i in range(assetN):
        xgb.fit(factRet,returns.iloc[:,i])
        mu.append(float(xgb.predict(f_bar)))
    mu=np.array(mu)
    Q = np.array(returns.cov())
    return mu,Q

def MaxSharpeRatio(Return,Cov,rf):
    n=len(Return)
    P=matrix(Cov)
    q=matrix(np.zeros((n, 1)))
    G=np.zeros((n,n))
    for i in range(n):
        G[i,i]=-1
    G=matrix(G)

    h=matrix(np.zeros((n, 1)))
    aaa=np.ones((1,n))
    A=matrix(np.vstack((aaa,Return)))
    
    given_r = []
    risk = []
    weight=[]    
    for temp_r in np.arange(max(min(Return),0),max(Return),0.0001):
        b=matrix(np.array([[1],[temp_r]]))
        # try:
        options['show_progress'] = False
        outcome = qp(P,q,G,h,A,b)
        x=np.array(outcome['x'])
        
        
        if outcome['status']!='optimal':
            # print(outcome['status'])
            continue
        given_r.append(temp_r)
        risk.append(sd(x.T,Cov))
        
        weight.append(x.round(4))
        # except:
            # continue        
    SharpeRatio=(np.array(given_r)-rf)/np.array(risk)
    return weight[SharpeRatio.argmax()]
        
def CVaR_Optimization(df,rf):
    inf=np.inf
    alpha=0.95
    Return=GeoMean(df)
    given_r=[]
    weight=[]
    CVaR=[]
    for temp_r in np.arange(max(0,min(Return)),max(Return),0.0001):
        # given_r=1.1*np.mean(Return)
        timeNum,assetNum=df.shape
        lb=np.concatenate((0*np.ones([1,assetNum]),np.zeros([1,timeNum]),0*np.ones([1,1])),axis=1).T
        bound=[]
        for i in range(len(lb)):
            temp_bound=(float(lb[i]),None)
            bound.append(temp_bound)
        
        aaa=np.concatenate((-df.values,-np.eye(timeNum),-np.ones([timeNum,1])),axis=1)
        bbb=np.concatenate((np.reshape(-Return,[1,len(Return)]),np.zeros([1,timeNum]),np.zeros([1,1])),axis=1)
        A = np.concatenate((aaa,bbb),axis=0)
        
        b=np.concatenate((np.zeros([timeNum,1]),temp_r*np.ones([1,1])),axis=0)
        
        Aeq=np.concatenate((np.ones([1,assetNum]),np.zeros([1,timeNum+1])),axis=1)
        beq=np.ones([1,1])
        
        k=1/((1-alpha)*timeNum)
        c=np.concatenate((np.zeros([assetNum,1]),k*np.ones([timeNum,1]),np.ones([1,1])),axis=0)
        
        outcomes=linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq,bounds=bound)
        if outcomes.success==True:
            weight.append(np.round(outcomes.x[0:assetNum],4))
            given_r.append(temp_r)
            CVaR.append(round(float(outcomes.fun),4))
    return weight,given_r,CVaR
        
AssetPr=pd.read_excel('Equity.xlsx',sheet_name='Adj Close',index_col=0)
FactorRe=pd.read_excel('Factor for test.xlsx',index_col=0)
# AssetPr['Date']=pd.to_datetime(AssetPr['Date'])

# AssetPr=AssetPr.set_index('Date').sort_index(ascending=True)
# date=AssetPr.tail(len(AssetPr)-1).index
# colNames=AssetPr.columns
# aaa=AssetPr.tail(len(AssetPr)-1).values/AssetPr.head(len(AssetPr)-1).values-1
AssetRe=pd.read_excel('Equity.xlsx',sheet_name='Return',index_col=0)
# AssetRe['Date']=pd.to_datetime(AssetRe['Date'])

# FactorRe['Date']=pd.to_datetime(FactorRe['Date'])
# FactorRe=FactorRe.set_index('Date').sort_index(ascending=True)

RiskFree=FactorRe['RF']
FactorRe=FactorRe.drop(columns=['RF'])

date=list(AssetPr.index)
index=list(range(len(date)))
Date2Index=dict(zip(date,index))
Index2Date=dict(zip(index,date))

TrainStart=0

InvestStart=pd.to_datetime('2015-01-31')
InvestStart=Date2Index[InvestStart]

RebalancePeriod=12

PortValue=[]
TimeSeries=[]
timeNum=0
InitialInvestment=1000000

while InvestStart<len(date):
    factor=FactorRe[TrainStart:InvestStart]
    assetRe=AssetRe[TrainStart:InvestStart]
    # assetPr=AssetPr.truncate(before=StartDate).truncate(after=EndDate)
    timeNum=assetRe.shape[0]
    # weight_CVaR,given_r,CVaR=CVaR_Optimization(assetRe,0.01)
    mu,Q=XGBoost(assetRe,factor)
    weight=MaxSharpeRatio(mu,Q,0.0005)
    
    assetPr=AssetPr.iloc[InvestStart].values
    money=weight*InitialInvestment
    amount=np.array([float(money[i])/float(assetPr[i]) for i in range(len(assetPr))])
    
    for t in range(RebalancePeriod):
        if InvestStart+t < len(AssetRe):
            print(InvestStart+t)
        # CalcuTime=RebalTime+relativedelta(months=t)
            assetPr=AssetPr.iloc[InvestStart+t].values    
            TimeSeries.append(Index2Date[InvestStart+t])
            PortValue.append(np.dot(amount.T,assetPr))
        
    TrainStart=TrainStart+RebalancePeriod
    InvestStart=InvestStart+RebalancePeriod

plt.plot(PortValue)

























