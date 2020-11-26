# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:42 2020

@author: hugha
"""
##################### upload the data ########################

import numpy as np
import pandas as pd
import os

path = "C:/Users/hugha/Documents/AI City MSc/intro to AI/CW/fer2013CW"

file_name = os.path.join(path, "fer2013.csv" )
df = pd.read_csv(file_name, na_values=['NA', '?'])

#this shuffles the data
df = df.reindex(np.random.permutation(df.index))

#drop the column Usage - don't know why its there or what it does
df.drop('Usage', 1, inplace=True)
print(df.columns)


#an example of test and train or K-fold on ex2part1



#adding a new column for each pixel
pixels_for_all_dpoints = []
for n in range(len(df['pixels'])):
    A = df['pixels'][n].split(' ')
    dpoint_pixels = []
    for i in A:
        dpoint_pixels.append(int(i))
    pixels_for_all_dpoints.append(dpoint_pixels)

pixels_for_all_dpoints = np.array(pixels_for_all_dpoints)

for n in range(len(pixels_for_all_dpoints[0])):
    df.insert(1, 'pixel'+str(n), pixels_for_all_dpoints[:,n])

print(df.columns)


#don't need this since now we have all those pixel columns
df.drop('pixels', 1, inplace=True)
print(df.columns)

#above here is all good

################ 1: what does the dataset look like? ############

print(df.columns)
print(df.shape)
print(df.size)

#are there any null values in my dataset. Q: are these only define by NA values when you import
print(df.isnull().any())


#making a smaller dataframe of only 10 datapoints to save time when 
#running code later on
df_quick = df[0:10]

print(df_quick.shape)
print(df_quick.size)

################### split the data #######################

result = []
for x in df_quick.columns:
    if x != 'emotion':
        result.append(x)

X = df_quick[result].values
y = df_quick['emotion'].values



# how does Kfold work?
#Kfold
from sklearn.model_selection import KFold

X_folds = []
y_folds = []

kf = KFold(4)

fold = 1
for train_index, validate_index in kf.split(X,y):
    trainDF = pd.DataFrame(df_quick.iloc[train_index, :])
    X_folds.append(trainDF)
    validateDF = pd.DataFrame(df_quick.iloc[validate_index])
    y_fold.append(validateDF)
    print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    fold += 1

####### this kfold chunk works #########











################ get the fucking dimensions down ################

# we want either Principal component analysis, or the nmf one

from sklearn.decomposition import PCA

pca = PCA(n_components=2)














################ Perceptron ###################

from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(4)

fold = 1
for train_index, validate_index in kf.split(df):
    trainDF = pd.DataFrame(df.iloc[train_index, :])
    validateDF = pd.DataFrame(df.iloc[validate_index])
    print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    fold += 1

#choose target as 'type'
result = []
for x in df.columns:
    if x != 'type':
        result.append(x)

X = df[result].values
y = df['type'].values


ppn = Perceptron(max_iter=20,tol=0.01,eta0=1)

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X,y):
    ppn.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = ppn.predict(X[validate_index])
    #print(y_test)
    #print(y_pred)
    #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1


#################### Questions ######################

#Do I store it in a table? I quite like a table


#what do I want to work out? could just run through the numbers and try everything that applies

#ex2part2 and ex3part1 have good postmodel visualisation stuff

#do you only use pd.DataFrame when loading datasets from sklearn