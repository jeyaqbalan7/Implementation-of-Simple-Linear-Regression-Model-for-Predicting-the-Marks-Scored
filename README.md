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
```
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

## Output:
![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/9b2e5b55-5a28-4107-8183-32a370a9183a)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/1fd35bd9-a94c-4699-a082-9e68e57d95c7)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/c135b4ce-f175-43ab-add8-65e891be1c80)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/f25b0e5c-113a-4b97-8fdc-4a2939b5f6b9)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/b6f43eea-0035-49b6-bec3-6583209a5322)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/5b7bca41-916c-4514-bcee-f14a5f326c5f)

![image](https://github.com/jeyaqbalan7/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393851/30a6202d-7050-47c8-b861-7eb53b98b092)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
