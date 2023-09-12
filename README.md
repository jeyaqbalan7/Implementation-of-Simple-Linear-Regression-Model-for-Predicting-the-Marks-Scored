# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the simple linear regression model for predicting the marks scored.

Step 2: Set variables for assigning dataset values and implement the .iloc module for slicing the values of the variables X and y.

Step 3: Import the following modules for linear regression; from sklearn.model_selection import train_test_split and also from sklearn.linear_model import LinearRegression.

Step 4: Assign the points for representing the points required for the fitting of the straight line in the graph.

Step 5: Predict the regression of the straight line for marks by using the representation of the graph.

Step 6: Compare the graphs (Training set, Testing set) and hence we obtained the simple linear regression model for predicting the marks scored using the given datas.

Step 7: End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jeyabalan
RegisterNumber: 212222240040
*/
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse
```

## Output:

## df.head()
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/c477c650-1ae8-400d-b051-e564c21da395)

## df.tail()
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/c3f55738-5274-4fe8-a3dd-90b6db16eadc)

## Array value of X
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/eae760d2-2953-4740-ac24-05b6e4972b2f)

## Array value of Y
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/c542f418-099a-46bf-bde2-f632498f8095)

## Values of Y prediction
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/68e47d45-1728-44ec-bb74-f4639611b0d3)

## Array values of Y test
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/8220824e-1e91-4b47-956c-e82343209944)

## Training Set Graph
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/55f14cc9-047d-4a31-af8b-f780dbcec8e7)

## Test Set Graph
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/2c9e5d1b-5d64-4abe-9715-42d12ee7f057)

## Values of MSE, MAE and RMSE
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/f209174e-a8f8-455c-8ee8-29bfa42d350f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
