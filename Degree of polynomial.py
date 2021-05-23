# -*- coding: utf-8 -*-
#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import sys
#%%

data=[]
with open("Gaussian_noise.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append(row)
    table=np.array(data)
print(table.shape)

#%% forming a label matrix
label=(table[:,1])
label=label.astype(np.float)
print(label.shape)
#%% forming a design matrix
design=table[:,0]
design=design.astype(np.float)
temp=[]
#%%Scatter plot
def scatter(table,size):
    label=(table[:,1])
    label=label.astype(np.float)
    design=table[:,0]
    design=design.astype(np.float)
    design=design[0:size]
    label=label[0:size]
    plt.scatter(design, label)
    plt.title('Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
#%%
scatter(table, 20)
#%% forming a function for design matrix
def form_design(a,table,size):
    design=table[0:size,0]
    design=design.astype(np.float)
    temp=[]
    for j in range (0,size):
        for i in range (0, a+1):
        
            temp.append(design[j]**i)
            final=np.array(temp)
    final=np.reshape(final,(size,a+1))
    return final


#%%Split a dataset into k folds
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

#%% findind the analytical solution using moore penrose inverse matrix
def get_theta_analytical(X, y, reg_param):
    return np.matmul(np.linalg.inv(np.matmul(X.T, X) + reg_param * np.identity(X.shape[1])),
                     np.matmul(X.T, y))
#%% inverse without regularixation
data_size=20
label=(table[:,1])
folds=4
label=label.astype(np.float)
e=[]
ya=[]
label=label[0:data_size]
error_label=[]
for a in range (0, 5):
    design_m=form_design(a,table,data_size)
    ya.append(a)
    x_cross,y_cross=cross_validation_split(design_m, label, folds)
    error=0
    for f in range(0,folds):
        x_cross_train=np.empty(shape=(0,a+1))
        y_cross_train=np.empty(shape=(0,1))
        x_cross_test=x_cross[f]
        y_cross_test=y_cross[f]
        for i in range(0,folds):
            if(i!=f):
                x_cross_train=np.concatenate((x_cross_train, x_cross[i]),axis=0)
                y_cross_train=np.concatenate((y_cross_train, y_cross[i]),axis=0)     
        w=get_theta_analytical(x_cross_train,y_cross_train,0)
        t=np.matmul(x_cross_test,w)
        error=error+np.sum((1/(2*len(y_cross_test)))*(t-y_cross_test)**2)
    error_label.append(error)
plt.plot(ya,error_label)
#%%inverse with regularization
#def minverse(table,data_size):
data_size=20
label=(table[:,1])
folds=4
label=label.astype(np.float)
e=[]
ya=[]
label=label[0:data_size]
lambda_p=np.linspace(0,0.1,num=100)
M=np.empty([len(lambda_p),10])
point=-1
for lamb in lambda_p:
    point+=1
    for a in range (0, 10):
        design_m=form_design(a,table,data_size)
        ya.append(a)
        x_cross,y_cross=cross_validation_split(design_m, label, folds)
        error=0
        for f in range(0,folds):
            x_cross_train=np.empty(shape=(0,a+1))
            y_cross_train=np.empty(shape=(0,1))
            x_cross_test=x_cross[f]
            y_cross_test=y_cross[f]
            for i in range(0,folds):
                if(i!=f):
                    x_cross_train=np.concatenate((x_cross_train, x_cross[i]),axis=0)
                    y_cross_train=np.concatenate((y_cross_train, y_cross[i]),axis=0)     
            w=get_theta_analytical(x_cross_train,y_cross_train,lamb)
            t=np.matmul(x_cross_test,w)
            error=error+np.sum((1/(2*len(y_cross_test)))*(t-y_cross_test)**2)
        M[point][a]=error
print(np.amin(M))
print(np.where(np.amin(M)==M))
    #plt.plot(ya,e)
    #plt.yscale("log")
    #errormin=np.array(e)
    #print(np.argmin(errormin))
#%%
#minverse(table,20)
#%% draw graph of hx
def graph_hx_inverse(table, data_size, a, lamb):
    label=(table[:,1])
    label=label.astype(np.float)
    label=label[0:data_size]
    design_m=form_design(a,table,data_size)
    error=0
    #w=np.matmul(np.linalg.inv(np.matmul(design_m.T, design_m)), np.matmul(design_m.T, label))
    w=get_theta_analytical(design_m,label,lamb)
    t=np.matmul(design_m,w)
    error=error+np.sum((t-label)**2)
    poly=0
    x = np.linspace(-1, 1.8, 1000)
    for i in range(0,a+1):
        poly=poly+w[i]*x**i
    plt.plot(x, poly,'g-')
    scatter(table, data_size)

#%%
graph_hx_inverse(table, 20,2,0.05)
#%% forming a label for appropriate degree of polynomial
t1=form_design(2,table,20)
#%% gradient descent
#def gradient_descent(table,degree,size):
"""
degree=8
size=100
design_g=form_design(degree,table,size)
label=(table[:,1])
label=label.astype(np.float)
wg=np.random.uniform(-15,15,size=(degree+1,1))
wnew=np.random.uniform(-1,1,size=(degree+1,1))
n=float(size)
alpha=0.001
errors=0
e=[]
prec=np.array([0.0000000000001]*(degree+1))
dw=np.array([1]*(degree+1))
label=np.reshape(label[0:size,],(size, 1))
times=0
ya=[]
while((dw>prec).all() and times<200000):
    wnew=wg+((alpha/n) * np.matmul(design_g.T, (label - np.matmul(design_g, wg))))
    dw=abs(wnew-wg)
    wg=wnew
    times=times+1
    print(times)
    t=np.matmul(design_g,wg)
    errors=np.sum((t-label)**2)
    e.append(errors)
    ya.append(times)
t=np.matmul(design_g,wg)
error2=np.sum((t-label)**2)
print (error2)
plt.plot(e,np.log(ya))
"""
#%%
#gradient_descent(table,8,100)
#%%
#def stochastic_gd(table,degree,size,minibatch_size):
degree=8
minibatch_size=50
size=100
lamb=0.066
label=(table[:,1])
label=label.astype(np.float)
label=label[0:size]
wg=np.random.uniform(-5,5,size=(degree+1,1))
wnew=np.random.uniform(-1,1,size=(degree+1,1))
n=float(size)
alpha=0.001
design_g=form_design(degree,table,size)
prec = np.empty([degree+1,1])
prec.fill(0.00000000000001)
dw=np.array([1]*(degree+1))
label=np.reshape(label[0:size,],(size, 1))
e=[]
y=[]
error_old=sys.maxsize
times=0
while( times<500000):#dw>prec).all() and
    error=0
    for i in range(0, design_g.shape[0], minibatch_size):
        X_train_mini = design_g[i:i + minibatch_size]
        y_train_mini = label[i:i + minibatch_size]
        wnew=wg*(1-(alpha*lamb/minibatch_size))+((alpha/minibatch_size) * np.matmul(X_train_mini.T, (y_train_mini - np.matmul(X_train_mini, wg))))
        dw=abs(wnew-wg)
        wg=wnew
        #if((dw<prec).all()):
        #    break
        t=np.matmul(design_g,wg)
        error=error+(1/(2*minibatch_size))*np.sum((t-label)**2)
    if(error>error_old):
        alpha=alpha/2
    error_old=error
    e.append(error)
    y.append(times)
    times=times+1
plt.yscale('log')
plt.plot((y),e)
t=np.matmul(design_g,wg)
error2=(1/(2*minibatch_size))*np.sum((t-label)**2)
print (error2)
#%%plotting graph
poly=0
x = np.linspace(-1, 1.8, 1000)
for i in range(0,degree+1):
    poly=poly+wg[i]*x**i
plt.plot(x, poly,'g-')
scatter(table, 100)
#%%
#%%
#def stochastic_gd_kfolds(table,degree,size,minibatch_size,folds):
"""
degree=6
size=20
minibatch_size=20
folds=4
label=(table[:,1])
label=label.astype(np.float)
label=label[0:size]
label=np.reshape(label[0:size,],(size, 1))
alpha=0.002
ya=[]
e=[]
error2=0
for a in range(0,degree+1):
    design_g=form_design(a,table,size)
    error=0
    ya.append(a)
    prec = np.empty([a+1,1])
    prec.fill(0.0000000000001)
    dw=np.array([1]*(a+1))
    x_cross,y_cross=cross_validation_split(design_g, label, folds)
    for f in range(0,folds):
        x_cross_train=np.empty(shape=(0,a+1))
        y_cross_train=np.empty(shape=(0,1))
        x_cross_test=x_cross[f]
        y_cross_test=y_cross[f]
        wg=np.random.uniform(-5,5,size=(a+1,1))
        wnew=np.random.uniform(-1,1,size=(a+1,1))
        for i in range(0,folds):
            if(i!=f):
                x_cross_train=np.concatenate((x_cross_train, x_cross[i]),axis=0)
                y_cross_train=np.concatenate((y_cross_train, y_cross[i]),axis=0)
        times=0
        while(times<100000):#(dw>prec).all() and 
            for i in range(0, x_cross_train.shape[0], minibatch_size):
                X_train_mini = x_cross_train[i:i + minibatch_size]
                y_train_mini = y_cross_train[i:i + minibatch_size]
                wnew=wg+((alpha/minibatch_size) * np.matmul(X_train_mini.T, (y_train_mini - np.matmul(X_train_mini, wg))))
                dw=abs(wnew-wg)
                wg=wnew
                t=np.matmul(x_cross_test,wg)
                #print(times,a)
                #print(np.sum((1/(2*minibatch_size))*(t-y_cross_test)**2))
                #if((dw<prec).all()):
                #   break
            times=times+1
            print(times,a)
        t=np.matmul(x_cross_test,wg)
        error=error+(1/(2*minibatch_size))*np.sum(*(t-y_cross_test)**2)
    e.append(error)
plt.plot(ya,e)
"""
#%%
#stochastic_gd_kfolds(table,5,20,1,4)
#%% with regularization
degree=10
size=100
minibatch_size=50
folds=4
label=(table[:,1])
label=label.astype(np.float)
label=label[0:size]
label=np.reshape(label[0:size,],(size, 1))
alpha=0.0003
ya=[]
e=[]
error2=0
lambda_p=np.linspace(0,0.1,num=10)
M=np.empty([len(lambda_p),(degree)])
point=-1
for lamb in lambda_p:
    point+=1
    for a in range(1,degree+1):
        design_g=form_design(a,table,size)
        error=0
        ya.append(a)
        prec = np.empty([a+1,1])
        prec.fill(0.000000000001)
        dw=np.array([1]*(a+1))
        x_cross,y_cross=cross_validation_split(design_g, label, folds)
        for f in range(0,folds):
            x_cross_train=np.empty(shape=(0,a+1))
            y_cross_train=np.empty(shape=(0,1))
            x_cross_test=x_cross[f]
            y_cross_test=y_cross[f]
            wg=np.random.uniform(-5,5,size=(a+1,1))
            wnew=np.random.uniform(-1,1,size=(a+1,1))
            derror=float(sys.maxsize)
            error_old=float(sys.maxsize)
            error_new=float(0)
            for i in range(0,folds):
                if(i!=f):
                    x_cross_train=np.concatenate((x_cross_train, x_cross[i]),axis=0)
                    y_cross_train=np.concatenate((y_cross_train, y_cross[i]),axis=0)
            times=0
            while(times<500000):# and (dw>prec).all() and (derror>0.000000001)):
                for i in range(0, x_cross_train.shape[0], minibatch_size):
                    X_train_mini = x_cross_train[i:i + minibatch_size]
                    y_train_mini = y_cross_train[i:i + minibatch_size]
                    wnew=wg*(1-(alpha*lamb/minibatch_size))+((alpha/minibatch_size) * np.matmul(X_train_mini.T, (y_train_mini - np.matmul(X_train_mini, wg))))
                    dw=abs(wnew-wg)
                    wg=wnew
                    t=np.matmul(x_cross_test,wg)
                    error_new=(1/(2*minibatch_size))*np.sum((t-y_cross_test)**2)
                    derror=(error_old-error_new)/error_old
                    #print(times,a)
                    #print(np.sum((1/(2*minibatch_size))*(t-y_cross_test)**2))
                    #if((dw<prec).all() or derror<0.00000000001):
                    #   break
                times=times+1
                #print(times,a,lamb)
            t=np.matmul(x_cross_test,wg)
            error=error+(1/(2*minibatch_size))*np.sum((t-y_cross_test)**2)
        print(lamb,a,error)
        M[point][a-1]=error
print(np.amin(M))
print(np.where(np.amin(M)==M))