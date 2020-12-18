# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:59:42 2020

@author: hugha
"""
##################### upload the data ########################
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
path = "C:/Users/hugha/OneDrive/Documents/AI City MSc/intro to AI/CW/fer2013CW"

file_name = os.path.join(path, "fer2013.csv" )
df = pd.read_csv(file_name, na_values=['NA', '?'])

#this shuffles the data
#df = df.reindex(np.random.permutation(df.index))

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
    df.insert(n, 'pixel'+str(n), pixels_for_all_dpoints[:,n])
# I need to flip the columns rounds so pixel-1 is at the start
#then pixel 2 etc.
print(df.columns)


#don't need this since now we have all those pixel columns
df.drop('pixels', 1, inplace=True)
print(df.columns)


################ 1: what does the dataset look like? ############

print(df.columns)
print(df.shape)
print(df.size)

#are there any null values in my dataset. Q: are these only define by NA values when you import
print(df.isnull().any())



################### split the data #######################

result = []
for x in df.columns:
    if x != 'emotion':
        result.append(x)

X = df[result].values
y = df['emotion'].values


############## One Hot Encoding


from keras.utils import to_categorical
y_cat = to_categorical(y)
y_cat[1]



################ get the fucking dimensions down ################

# we want either Principal component analysis, or the nmf one

#ontop of PCA use t-SNE
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA

pca = PCA(n_components=2304) 
pca.fit(X)

pca.explained_variance_ratio_
cummulative_variance=np.cumsum(pca.explained_variance_ratio_)
plt.step(range(cummulative_variance.size), cummulative_variance)
# with th graph showing cumulative variance, we can see how much variance loss 
#we get when we drop dimensions. 
#can see that we get 90% of the variance up to about 500

pca_used = PCA(n_components=1600) #cut this down to 128
pca_used.fit(X)
X_pca = pca_used.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
# new base only size 10, but don't panic Hugh, as I run it with more sample,
#then 'n_components' can go up and it will have been reduced to a reasonable
# size
X_reshaped = []
for datapoint in X_pca:
    X_reshaped.append(datapoint.reshape(40,40))
X_CNN_train = np.array(X_reshaped)
#X_CNN_train = np.expand_dims(X_CNN_train, axis=3)
print(X_CNN_train.shape)
print(X_CNN_train[0][0])
print(X_CNN_train[0].shape)
plt.imshow(X_CNN_train[100])

# size
X_reshaped = []
for datapoint in X:
    X_reshaped.append(datapoint.reshape(48,48))
X_CN_train = np.array(X_reshaped)
plt.imshow(X_CN_train[1])
#get down to three components and plot it
#re-train data with 3 dimensions

from sklearn.decomposition import NMF
nmf = NMF(n_components = 1600)
X_nmf = nmf.fit_transform(X_test)

plt.imshow(X[1])

#get matplotlib to show images before and after 

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size = 0.2, random_state = 42)


################ Neural Network ###################

# what are the answers to 'basic first cut decisions' when working with 
#facial recognition data, classification, and NN MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.1)
optim = optimizers.Adam(learning_rate = lr_schedule)
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(128, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(len(y_train[1]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optim)
model.summary()

model.fit(X_train,y_train,verbose=1,epochs=10, batch_size = 32)
#this gives me a loss of 1.16
pred_hot = model.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#calculate accuracy
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {}".format(score))


model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(len(y_train[1]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])
model.summary()

model.fit(X_train,y_train,verbose=1,epochs=30)
#this gives me a loss of 1.16

pred_hot = model.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#calculate accuracy
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {}".format(score))

#this generalises very badly
#dropout



model1 = Sequential()
model1.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid')) # Hidden 1
model1.add(Dense(len(y_train[1]), activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model1.summary()

model1.fit(X_train,y_train,verbose=1,epochs=50)
#without PCA gives me a loss of 1.8

pred_hot = model1.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#calculate accuracy
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

########## comments on NN ################
#this network works v bad with relu, stops at 3

######### NN with two classes ############


# how many points in each class
classes = []
for target in range(7):
    point_class = []
    for point_index in range(len(y)):
        if y[point_index] == target:
            point_class.append(point_index)
    classes.append(point_class)
    print(len(classes[target]))

print(len(classes[0]))



########################## OVR ##############################

y_zero = [1 if x == 0 else 0 for x in y]
y_one = [1 if x == 1 else 0 for x in y]
y_two = [1 if x == 2 else 0 for x in y]
y_three = [1 if x == 3 else 0 for x in y]
y_four = [1 if x == 4 else 0 for x in y]
y_five = [1 if x == 5 else 0 for x in y]
y_six = [1 if x == 6 else 0 for x in y]


#Anger
y_zero[0]
plt.imshow(X[0].reshape(48,48))
#Disgust
y_one[299]
plt.imshow(X[299].reshape(48,48))
#fear
y_two[95:96]
plt.imshow(X[96].reshape(48,48))
#happy
y_three[8]
plt.imshow(X[8].reshape(48,48))
#sad
y_four[:10]
plt.imshow(X[6].reshape(48,48))
plt.imshow(X[6].reshape(48,48))

#surprise
y_five[15]
plt.imshow(X[15].reshape(48,48))
#neutral
y_six[4]
plt.imshow(X[4].reshape(48,48))

##############################################
#choose class 3 and 4 because these have most data points
two_class_index = []
for point_index in range(len(y)):
    if y[point_index] == 3:
        two_class_index.append(point_index)
#for point_index in range(len(y)):        
    if y[point_index] == 4:
        two_class_index.append(point_index)

print(len(two_class_index))

X_two_class = X[two_class_index]
y_two_class = y[two_class_index]

y_two_class = [0 if x == 3 else 1 for x in y_two_class]
y_two_class_cat = to_categorical(y_two_class)

pca = PCA(n_components=2304) 
pca.fit(X_two_class)

pca.explained_variance_ratio_
cummulative_variance=np.cumsum(pca.explained_variance_ratio_)
plt.step(range(cummulative_variance.size), cummulative_variance)

pca_used = PCA(n_components=484) 
pca_used.fit(X_two_class)
X_two_class_pca = pca_used.transform(X_two_class)
print("original shape:   ", X_two_class.shape)
print("transformed shape:", X_two_class_pca.shape)

X_two_train, X_two_test, y_two_train, y_two_test = train_test_split(X_two_class_pca, y_two_class_cat, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Dense(64, input_dim=X_two_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, input_dim=X_two_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, input_dim=X_two_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, input_dim=X_two_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(32, input_dim=X_two_train.shape[1], activation='sigmoid'))
model.add(Dense(len(y_two_train[1]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam')
model.summary()
model.fit(X_two_train,y_two_train,verbose=1,epochs=3, batch_size = 30)
pred_hot = model.predict(X_two_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_two_test,axis=1) 
#calculate accuracy
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {} %".format(score*100))

############### CNN #####################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, SeparableConv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

#X needs to be a tensor
X_reshaped = []
for datapoint in X_train:
    X_reshaped.append(datapoint.reshape(22,22))
X_CNN_train = np.array(X_reshaped)
X_CNN_train = np.expand_dims(X_CNN_train, axis=3)
print(X_CNN_train.shape)
print(X_CNN_train[0][0])
print(X_CNN_train[0].shape)

X_test_reshaped = []
for datapoint in X_test:
    X_test_reshaped.append(datapoint.reshape(22,22))
X_CNN_test = np.array(X_test_reshaped)
X_CNN_test = np.expand_dims(X_CNN_test, axis=3)
print(X_CNN_test.shape)
print(X_CNN_test[0][0])
print(X_CNN_test[0].shape)

#datagen.fit(X_CNN_train)

num_classes = y_train.shape[1]
save_dir = './' 
model_name = 'Hughs CNN'

model2 = Sequential()
model2.add(Conv2D(8, kernel_size=(1, 1), activation='relu', padding = 'same', input_shape= X_CNN_train[0].shape))
model2.add(BatchNormalization(axis=1))
model2.add(Conv2D(8, kernel_size=(1, 1), activation='relu', padding = 'same', input_shape= X_CNN_train[0].shape))
model2.add(BatchNormalization(axis=1))

model2.add(SeparableConv2D(16, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(16, kernel_size=(3, 3), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(16, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

model2.add(SeparableConv2D(32, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(32, kernel_size=(3, 3), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(32, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

model2.add(SeparableConv2D(64, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(64, kernel_size=(3, 3), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(64, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

model2.add(SeparableConv2D(128, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(128, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

model2.add(SeparableConv2D(256, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(256, kernel_size=(3, 3), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(SeparableConv2D(256, kernel_size=(1, 1), activation='relu'))
model2.add(BatchNormalization(axis=1))
model2.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

#model2.add(Flatten())
model2.add(Dense(num_classes))
model2.add(GlobalAveragePooling2D())
model2.add(Activation('softmax'))



model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.summary()
#this code works, I need to tweak with parameters

model2.fit(X_CNN_train,y_train,verbose=2,epochs=30, batch_size = 128)

pred_hot = model2.predict(X_CNN_test)
#now pick the most likely outcome
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#calculate accuracy
from sklearn import metrics
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {} %".format(score*100))
###################################################### Ugly Duckling



#################### K-means #########################

from sklearn.cluster import KMeans

kmeans = KMeans(7, random_state = 0)
labels = kmeans.fit(X_train).predict(X_test)



def accuracy_score(y_pred, y):
     accuracy_sum = 0
     for datapoint in range(len(y)):
         if np.argmax(y[datapoint]) == np.argmax(y_pred[datapoint]):
             accuracy_sum += 1
    
     accuracy = accuracy_sum/len(y)
     return accuracy*100
kmeans_score = accuracy_score(y_test, labels)

print("Accuracy score: {}".format(kmeans_score))
labels[0]
#accuracy of 13.8, basically random

###################

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(X_two_test)
labels_gmm = gmm.predict(X_two_test)
gmm_score = metrics.accuracy_score(y_two_test, labels_gmm)
print("Accuracy score: {}".format(gmm_score))





#################### Questions ######################

a = np.arange(16)
print(a)
b = a.reshape(4,4)
print(b)
#what do I want to work out? could just run through the numbers and try everything that applies

#ex2part2 and ex3part1 have good postmodel visualisation stuff

