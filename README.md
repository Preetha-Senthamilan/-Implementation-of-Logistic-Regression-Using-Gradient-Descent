# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.

6.Obtain the graph.
## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:PREETHA.S 
RegisterNumber: 212222230110 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:

Array value of X:

![Screenshot 2023-10-10 093209](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/7bbc4b28-74cb-48b5-92ba-e98bc222d1fc)


Array value of Y:

![Screenshot 2023-10-10 094235](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/c0a96198-9fad-4fe8-964c-64fd851a1257)


Exam 1 score graph:

![Screenshot 2023-10-10 133427](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/7c00933a-2321-430f-9b43-191246721a9c)

Sigmoid function graph:

![Screenshot 2023-10-10 133439](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/dfa7ce3e-d193-41b3-bc32-f115eec2f24a)

X_Train_grad value: 

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/c893fb16-de5e-49c4-8919-8affc63f1e71)


Y_Train_grad value:

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/a43f0b6d-c7f9-457a-8696-d425d6b5a20d)


Print res.X:

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/ba0736f4-1a1a-4c56-9336-bf73db7d41dc)


Decision boundary-gragh for exam score:

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/3d314673-a217-46d8-9ceb-d239261550fb)


Probability value:

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/70dc9b28-d232-47cf-b3b5-2e89422571d8)


Prediction value of mean:

![image](https://github.com/Preetha-Senthamilan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119390282/6d48c3c6-3361-4e54-baf6-61563f61813b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

