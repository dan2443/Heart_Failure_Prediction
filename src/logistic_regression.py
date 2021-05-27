# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = dataset.iloc[:, :-2].values # Removing time (not necessary) and result
y = dataset.iloc[:, -1].values # Keeping only result

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
# Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
X_train = scaler_model.fit_transform(X_train)
X_test = scaler_model.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Test set results
f, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot = True, fmt='.0f', ax = ax)
plt.title('Confusion Matrix Logistic Regression')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Accuracy of Test set
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
mylist = []
ac_test = accuracy_score(y_test, y_pred)
mylist.append(ac_test)
print('Accuracy testing:', np.round(ac_test,3))
print(classification_report(y_test, y_pred))
