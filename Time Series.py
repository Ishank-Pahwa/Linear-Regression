#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange
import datetime
import sys
#%%
data=[]
a=0
with open("train.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if(a==0):
            a=1
            continue;
        data.append(row)
    table=np.array(data)
print(table.shape)       

#%% Get complete set of X and y values
dates = np.array([datetime.datetime.strptime(i[0], '%m/%d/%y') for i in data])
values = np.array([float(i[1]) for i in data])

# Sort the values
values = values[np.argsort(dates)]
dates = np.sort(dates)

#%% Plot the values
#pred = [(15 * np.sin(i.month * (np.pi / 6) - np.pi / 2) + 10) for i in dates]
plt.plot(dates,values)
#plt.show()
#%%
month_format=[]
for i in dates:
    month_format.append(i.month)
#%%
month=[]
for i in dates:
    month.append(i.month+i.year*12-24059)
plt.plot(month,values)
#%%
def cross_validation_split(X, y, folds):
    X_split = list()
    y_split = list()
    X_copy = list(X)
    y_copy = list(y)
    fold_size = int(len(X) / folds)
    # Create splits now :
    for i in range(folds):
        X_fold = list()
        y_fold = list()
        while len(X_fold) < fold_size:
            index = randrange(len(X_copy))
            X_fold.append(X_copy.pop(index))
            y_fold.append(y_copy.pop(index))
        X_split.append(np.array(X_fold))
        y_split.append(np.reshape(np.array(y_fold), (fold_size,1)))
    return X_split, y_split
#%%
def design_mat(size,month_format, order,dates,factor):
    months=np.array(month_format)
    months=months.astype(np.float)
    temp=[]
    for j in range(0,size):
        for sc in range(0,2):
            for i in range(0,order+1):
                if(i==0):
                    if(sc==1):
                        continue
                    temp.append(1)
                else:
                    if(sc==0):
                        temp.append(np.sin((months[j]+(dates[j].year-2004)*12)*i*np.pi/factor))
                    else:
                        temp.append(np.cos((months[j]+(dates[j].year-2004)*12)*i*np.pi/factor))
                        
    final=np.array(temp)
    final=np.reshape(final,(size,(2*order+1)))
    add_year=np.full([size,10],0)
    count=0
    for i in dates:
        if(i.year-2004!=10):
            add_year[count][i.year-2004]=1
            count+=1
    #final=np.concatenate((final,add_year), axis=1)
    return final
#%%
#design_g=design_mat(month_format,11)
#%%
f_range=np.linspace(4,24,num=6)
for f in f_range:
    for a in range(2,9):
        design_g=design_mat(110,month_format,a,dates,f)
        degree=a
        size=110
        wg=np.random.uniform(-15,15,size=(2*degree+1,1))
        wnew=np.random.uniform(-1,1,size=(2*degree+1,1))
        n=float(size)
        alpha=1
        errors=0
        e=[]
        prec=np.array([0.00000000000000000001]*(2*degree+1))
        dw=np.array([1]*(2*degree+1))
        values=np.reshape(values[0:size,],(size, 1))
        times=0
        label=values
        error_old=sys.maxsize
        ya=[]
        while(times<500000 and (dw>prec).all() ):
            wnew=wg+((alpha/n) * np.matmul(design_g.T, (label - np.matmul(design_g, wg))))
            dw=abs(wnew-wg)
            wg=wnew
            times=times+1
            #print(times)
            t=np.matmul(design_g,wg)
            errors=np.sum((t-label)**2)
            if(errors>error_old):
                alpha=alpha/2
            error_old=errors
            e.append(errors)
            ya.append(times)
        t=np.matmul(design_g,wg)
        error2=np.sum((t-label)**2)
        print (f, a, error2/110)
#plt.plot(e,np.log(ya))
#%%
a=8
f=24
design_g=design_mat(110,month_format,a,dates,f)
degree=a
size=110
wg=np.random.uniform(-15,15,size=(2*degree+1,1))
wnew=np.random.uniform(-1,1,size=(2*degree+1,1))
n=float(size)
alpha=1
errors=0
e=[]
prec=np.array([0.00000000000000000001]*(2*degree+1))
dw=np.array([1]*(2*degree+1))
values=np.reshape(values[0:size,],(size, 1))
times=0
label=values
error_old=sys.maxsize
ya=[]
while(times<500000 and (dw>prec).all() ):
    wnew=wg+((alpha/n) * np.matmul(design_g.T, (label - np.matmul(design_g, wg))))
    dw=abs(wnew-wg)
    wg=wnew
    times=times+1
    #print(times)
    t=np.matmul(design_g,wg)
    errors=np.sum((t-label)**2)
    if(errors>error_old):
        alpha=alpha/2
    error_old=errors
    e.append(errors)
    ya.append(times)
t=np.matmul(design_g,wg)
error2=np.sum((t-label)**2)
print (f, a, error2/110)
#%% Test Data
test = []
with open('./test.csv','r')as f:
  csv_reader = csv.reader(f)
  for line in csv_reader:
      test.append(line)
     
test = test[1:]   

dates_test = np.array([datetime.datetime.strptime(i[0], '%m/%d/%y') for i in test])
month_format_test=[]
for i in dates_test:
    month_format_test.append(i.month)
test_design=design_mat(10,month_format_test,a,dates_test,24)
#%%
ans=np.matmul(test_design,wg)
#%%
plt.scatter(dates,values)
plt.scatter(dates_test,ans)
plt.plot(dates,t)
#%%
#final_ans=np.concatenate((test,ans),axis=1)
#label_ans=['id','value']
#final_ans=np.concatenate(label_ans,final_ans)
#pd.DataFrame(final_ans).to_csv('test_ans.csv',header=label_ans)
test_index = [i[0] for i in test]
t = pd.DataFrame(ans, index = test_index)
t.to_csv("submission.csv")


