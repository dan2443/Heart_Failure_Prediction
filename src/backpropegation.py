# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dataset.isnull().sum()
X = dataset.iloc[:, :11].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# (input + output / 2) = units in hidden layers
classifier.add(Dense(units = 6, activation='relu', kernel_initializer='uniform',input_dim=11))

# Adding second hidden layer
classifier.add(Dense(units = 6, activation='relu', kernel_initializer='uniform'))

# Adding output layer
classifier.add(Dense(units = 1, activation='sigmoid', kernel_initializer='uniform'))

# Compiling the ANN
# adam - algorithem to optimize the weights
# binary_crossentropy - loss function for binary values (like logistic regression)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch size = number of observations after which you wanna update the weights
# epochs = number of times you train the whole ann
classifier.fit(X_train, y_train,batch_size = 10, epochs = 300)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# changing the result to True or False base on our threshold = 0.5
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Test set results
f, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot = True, fmt='.0f', ax = ax)
plt.title('Confusion Matrix Backpropagation')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
