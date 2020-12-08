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


#making a smaller dataframe of only 10 datapoints to save time when 
#running code later on
df_quick = df[0:100]

print(df_quick.shape)
print(df_quick.size)

################### split the data #######################

result = []
for x in df.columns:
    if x != 'emotion':
        result.append(x)

X = df[result].values
y = df['emotion'].values

#y = one_hot_encoding(y)

from keras.utils import to_categorical
y = to_categorical(y)
y[1]

# how does Kfold work? use index to call X and y, X[train_index] in the for loop
#Kfold
#from sklearn.model_selection import KFold

#training_folds = []
#validate_folds = []

#kf = KFold(4)

#fold = 1
#for train_index, validate_index in kf.split(X,y):
#    trainDF = pd.DataFrame(df_quick.iloc[train_index, :])
#    training_folds.append(trainDF)
#    validateDF = pd.DataFrame(df_quick.iloc[validate_index])
#    validate_folds.append(validateDF)
#    print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
#    fold += 1

#print(training_folds[1]) # use this for now for comp ease
####### this kfold chunk works #########


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.2,
    height_shift_range=0.2, brightness_range=None, shear_range=0.2, zoom_range=0.2,
    channel_shift_range=0.2, fill_mode='nearest', cval=0.0, horizontal_flip=True,
    vertical_flip=True, rescale=None, preprocessing_function=None,
    data_format=None, validation_split=0.0, dtype=None
)







################ get the fucking dimensions down ################

# we want either Principal component analysis, or the nmf one

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

pca = PCA(n_components=2304) 
pca.fit(X)

pca.explained_variance_ratio_
cummulative_variance=np.cumsum(pca.explained_variance_ratio_)
plt.step(range(cummulative_variance.size), cummulative_variance)
# with th graph showing cumulative variance, we can see how much variance loss 
#we get when we drop dimensions. 
#can see that we get 90% of the variance up to about 500

pca_used = PCA(n_components=484) 
pca_used.fit(X)
X_pca = pca_used.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
# new base only size 10, but don't panic Hugh, as I run it with more sample,
#then 'n_components' can go up and it will have been reduced to a reasonable
# size



X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)


######## NMF #######

nmf = NMF(n_components=1000)
nmf.fit(X)
X_nmf = nmf.transform(X)
#this just bugs out

######### Isomap
from sklearn.manifold import Isomap
embedding = Isomap(n_components=500)
X_iso = embedding.fit_transform(X[:100])


################ Neural Network ###################

# what are the answers to 'basic first cut decisions' when working with 
#facial recognition data, classification, and NN MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9)
optim = optimizers.Adam(learning_rate = lr_schedule)
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, input_dim=X_train.shape[1], activation='sigmoid'))
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

############### CNN #####################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#X needs to be a tensor
X_reshaped = []
for datapoint in X_train:
    X_reshaped.append(datapoint.reshape(22,22))
X_CNN_train = np.array(X_reshaped)
X_CNN_train = np.expand_dims(X_CNN_train, axis=3)
print(X_CNN_train.shape)
print(X_CNN_train[0][0])
print(X_CNN_train[0].shape)

datagen.fit(X_CNN_train)

num_classes = y.shape[1]
save_dir = './' 
model_name = 'Hughs CNN'

model2 = Sequential()
model2.add(Conv2D(64, kernel_size=(4, 4), activation='relu', input_shape= X_CNN_train[0].shape))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(128, activation='sigmoid'))

model2.add(Dense(num_classes))
model2.add(Activation('softmax'))



model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.summary()
#this code works, I need to tweak with parameters

model2.fit(X_CNN_train,y_train,verbose=2,epochs=24)

#################### Questions ######################

a = np.arange(16)
print(a)
b = a.reshape(4,4)
print(b)
#what do I want to work out? could just run through the numbers and try everything that applies

#ex2part2 and ex3part1 have good postmodel visualisation stuff

