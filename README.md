# heart_failure_prediction
In part of Machine learning seminar, our goal was to analyze a dataset with machine learning algoritems to predict an outcome.
Me and my partner decided to take this dataset from Kaggle:

https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

## Introduction:

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.


## Machine Learning algorithms used in this project:

* Logistic Regression
* Backpropagation
* Naive Bayes Classifier

### Logistic Regression:

Logistic Regression is a statistical model which used to find the probability of an appearance, in our case the probability of our patient to die by heart faliure.
The Logistic Regression algoritem is based on Linear Regression but improved by using Sigmoid and ln functions.

### Backpropagation

Backpropagation is an algorithm which built on ANN (artificial neural network), in our implementation we used Rectifier Linear and Sigmoid function to predict our outcome.

### Naive Bayes Classifier

Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.
Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.

## Corralation between parameters

### Corralation between all parameters

![Corralation between parameters](https://github.com/dan2443/heart_failure_prediction/blob/main/images/parameters%20correlation.png)

### Corralation between DEATH_EVENT parameter to most significant parameters

![Corralation with threshold](https://github.com/dan2443/heart_failure_prediction/blob/main/images/parameters%20correlation%20with%20threshold.png)


## Results

### Logistic Regression
Results using normalization technique:

![Logistic Regression normalization](https://github.com/dan2443/heart_failure_prediction/blob/main/images/logistic%20regression%20normalization.png)

Results using standartization technique:

![Logistic Regression standartization](https://github.com/dan2443/heart_failure_prediction/blob/main/images/logistic%20regression%20standartization.png)

### Backpropagation

![Backpropagation results](https://github.com/dan2443/heart_failure_prediction/blob/main/images/Backpropegation%20results%20300epochs.png)

### Naive Bayes Classifier

[Naive Bayes Classifier results](https://github.com/dan2443/heart_failure_prediction/blob/main/images/Naive%20Bayes%20results.png)

### Backpropagation using only the most significant parameters

[Backpropagation results with significant](https://github.com/dan2443/heart_failure_prediction/blob/main/images/Backpropegation%20results%20150epochs%20best%20param%201%20hidden.png)

### Results of all algorithms

[Results of all algorithms](https://github.com/dan2443/heart_failure_prediction/blob/main/images/results%20of%20all%20algorithms.png)



