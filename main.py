# ---------------------------------------------------- #
# Import classes and functions
# ---------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense
import keras

# ---------------------------------------------------- #
# data preparation
# ---------------------------------------------------- #
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)

n_features = X.shape[1]
n_classes = Y.shape[1]


from time import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard
from datetime import datetime
from packaging import version

model = Sequential()

model.add(Dense(10, input_shape=(n_features,), activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

tensorboard = TensorBoard(
	log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
	histogram_freq=0, 
	write_graph=True, 
	write_images=True)

model.fit(
	X_train, 
	Y_train, 
	verbose=1, 
	epochs=50000,
	#validation_split=0.25,
	validation_data=(X_test, Y_test),
	callbacks=[tensorboard]
	)

