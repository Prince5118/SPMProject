# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:24:41 2020

@author: sachi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

#function for predicting testset accuracy
def accuracy(y,pred):
	count=0.0
	for i in range(0,y.shape[0]):
		if(y[i][0]==pred[i]):
			count=count+1
	return count*100/y.shape[0]

#importing the dataset
X=pd.read_csv("reduced_features.csv")
X=X.iloc[:,:].values
Y=pd.read_csv("target_output.csv")
Y=Y.iloc[:,:].values


#encoding categorical data


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.03, random_state=23)

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
Y_train=onehotencoder.fit_transform(Y_train).toarray()

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initialising the ann
classifier= Sequential()


#adding input layer and 1st hidden layer
classifier.add(Dense(output_dim=64,activation='relu',init='uniform',input_shape=(175,)))

# adding droput to hidden layer 1
classifier.add(Dropout(0.3))

#adding second hidden layer
classifier.add(Dense(output_dim=32,activation='relu',init='uniform'))

#adding droput to hidden layer 2
classifier.add(Dropout(0.3))

#adding third hidden layer
classifier.add(Dense(output_dim=16,activation='relu',init='uniform'))

#adding droput to hidden layer 3
classifier.add(Dropout(0.3))



#adding output layer
classifier.add(Dense(output_dim=13,activation='softmax',init='uniform'))


#compiling ANN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
history = classifier.fit(X_train,Y_train,batch_size=32,epochs=500, verbose=1)


from keras.utils import plot_model
plot_model(classifier, to_file='model.png', show_shapes=True)


#predicting test sets results
Y_pred=classifier.predict(X_test)
Y_test_op=((np.argmax(Y_pred,axis=1)+1))
np.reshape(Y_test_op,(Y_pred.shape[0],1))
print(accuracy(Y_test,Y_test_op))


#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_test_op)




# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

