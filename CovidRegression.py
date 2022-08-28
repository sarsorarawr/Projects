#!/usr/bin/env python
# coding: utf-8

# ### 1. The file CaCovidInfMarch24toMidJuly.txt on class website contains daily new cases of Covid-19 in CA from March 24 to mid July 2020, for a period of 120 days.  Use the first 90 days for training a linear regression model ( I = A + B t ) to predict the infected cases the next 30 days.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

covid = pd.read_csv('CaCovidInfMarch24toMidJuly.txt',header=None)
X=np.arange(90).reshape((90, -1))
y=covid[:90]
y=np.array(y)


# #### a) Compute optimal solution by solving normal equation

# In[2]:


X_b = np.c_[np.ones((90, 1)), X]

#solving coefficients from normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# In[5]:


#coefficients from Sk-learn

print(regressor.intercept_)
print(regressor.coef_)


# In[6]:


#Mean squared error
X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 1)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta_best)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[7]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=np.array(covid[:])
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta_best[1]*xx+theta_best[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta_best[1]*xxx+theta_best[0]
plt.plot(xxx, yyy, "g--")
plt.title('Normal Least Squared')


# #### b) Compute optimal solution by full batch gradient descent

# In[8]:


eta = 0.000005 # step size
n_itr = 90000
theta = np.random.randn(2,1)

for itr in range(n_itr):
    grad = X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * grad
theta


# In[9]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 1)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[10]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('Batch Gradient Descent')


# #### c) Compute optimal solution by mini-batch (size 10) stochastic gradient descent

# In[11]:


n_epoch = 3000
minibatch_size = 10 # number of data point in a batch
lr = 0.0005
wgt_decay = 0.9

theta = np.random.randn(2,1)
#theta = np.array([[500],[3]])
theta_0_path = [theta[0]]
theta_1_path = [theta[1]]

for epoch in range(n_epoch):
    if epoch%500 == 0:
        plt.plot(xaxis,xaxis*theta[0]+theta[1],'g-.')
    # shuffle/randomize indices towards iid samples
    # to improve settling down towards convergence    
    shuffled_indices = np.random.permutation(90) # shuffleed indices of the number of data points
    Xb_shuffled = X_b[shuffled_indices] 
    y_shuffled = y[shuffled_indices]
    lr=0.0005
    for i in range(0, 90, minibatch_size):
        Xi = Xb_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        grad = Xi.T.dot(Xi.dot(theta)-yi) * (2/minibatch_size)
        theta = theta - lr*grad
        theta_0_path.append(theta[0])
        theta_1_path.append(theta[1])
        lr *= wgt_decay # update lr
    
print(theta)
# plotting
plt.plot(X,y,'k.')
plt.plot(xaxis,yaxis,'b--')
plt.plot(xaxis,xaxis*theta[0]+theta[1],'r-')
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18) 
plt.show()


# In[12]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 1)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[13]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('mini-batch SGD')


# ### 2.  Repeat Problem 1 for quadratic polynomial regression model ( I = A + B t + C t 2 ).

# #### a) Compute optimal solution by solving normal equation

# In[14]:


X_b = np.c_[np.ones((90, 1)), X, X**2]

#solving coefficients from normal equation
theta_est = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_est)


# In[15]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 2)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta_est)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[16]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta_est[2]*xx**2+theta_est[1]*xx+theta_est[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta_est[2]*xxx**2+theta_est[1]*xxx+theta_est[0]
plt.plot(xxx, yyy, "g--")
plt.title('Least squared quadratic')


# #### b) compute optimal solution by full batch gradient descent

# In[17]:


eta = 1e-12 # step size
n_itr = 5000
#theta = np.random.randn(3,1)
theta = np.array([[100],[2],[0]])
for itr in range(n_itr):
    grad = X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * grad

theta


# In[18]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 2)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[19]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[2]*xx**2+theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[2]*xxx**2+theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('Full Batch Gradient Descent Quadratic')


# #### c) compute optimal solution by mini-batch (size 10) stochastic gradient descent

# In[27]:


n_epoch = 4000
minibatch_size = 10 # number of data point in a batch
lr = 5e-14
wgt_decay = 0.9

theta = np.random.randn(3,1)
#theta = np.array([[100],[2],[0]])
theta_0_path = [theta[0]]
theta_1_path = [theta[1]]
theta_2_path = [theta[2]]

for epoch in range(n_epoch):
    if epoch%500 == 0:
        plt.plot(xaxis,xaxis**2*theta[0]+xaxis*theta[1]+theta[2],'g-.')
    # shuffle/randomize indices towards iid samples
    # to improve settling down towards convergence    
    shuffled_indices = np.random.permutation(90) # shuffleed indices of the number of data points
    Xb_shuffled = X_b[shuffled_indices] 
    y_shuffled = y[shuffled_indices]
    lr=5e-14
    for i in range(0, 90, minibatch_size):
        Xi = Xb_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        grad = Xi.T.dot(Xi.dot(theta)-yi) * (2/minibatch_size)
        theta = theta - lr*grad
        theta_0_path.append(theta[0])
        theta_1_path.append(theta[1])
        theta_2_path.append(theta[2])
        lr *= wgt_decay # update lr
    
print(theta)
# plotting
plt.plot(X,y,'k.')
plt.plot(xaxis,yaxis,'b--')
plt.plot(xaxis,xaxis**2*theta[0]+xaxis*theta[1]+theta[2],'r-')
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18) 
plt.show()


# In[28]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 2)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[29]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[2]*xx**2+theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[2]*xxx**2+theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('mini-batch Stochastic Gradient Descent Quadratic')


# ### 3. Repeat Problem 1 for the cubic polynomial regression model (  I = A + B t + C t 2 + D t 3 )

# #### a) Compute optimal solution by solving normal equation

# In[30]:


X_b = np.c_[np.ones((90, 1)), X, X**2,X**3]

#solving coefficients from normal equation
theta_est = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_est)


# In[31]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 3)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta_est)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[32]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta_est[3]*xx**3+theta_est[2]*xx**2+theta_est[1]*xx+theta_est[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta_est[3]*xxx**3+theta_est[2]*xxx**2+theta_est[1]*xxx+theta_est[0]
plt.plot(xxx, yyy, "g--")
plt.title('Least squared cubic')


# #### b) compute optimal solution by full batch gradient descent

# In[33]:


eta = 1e-20 # step size
n_itr = 200000
#theta = np.random.randn(4,1)
theta = np.array([[700],[50],[-9e-1],[1e-2]])
for itr in range(n_itr):
    grad = X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * grad

theta


# In[34]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 3)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[35]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[3]*xx**3+theta[2]*xx**2+theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[3]*xxx**3+theta[2]*xxx**2+theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('Full batch gradient descent cubic')


# #### c) compute optimal solution by mini-batch (size 10) stochastic gradient descent

# In[93]:


n_epoch = 5000
minibatch_size = 10 # number of data point in a batch
lr = 5e-14
wgt_decay = 0.9

theta = np.random.randn(4,1)
theta_0_path = [theta[0]]
theta_1_path = [theta[1]]
theta_2_path = [theta[2]]
theta_3_path = [theta[3]]

for epoch in range(n_epoch):
    if epoch%500 == 0:
        plt.plot(xaxis,xaxis**3*theta[0]+xaxis**2*theta[1]+xaxis*theta[2]+theta[3],'g-.')
    # shuffle/randomize indices towards iid samples
    # to improve settling down towards convergence    
    shuffled_indices = np.random.permutation(90) # shuffleed indices of the number of data points
    Xb_shuffled = X_b[shuffled_indices] 
    y_shuffled = y[shuffled_indices]
    lr=5e-14
    for i in range(0, 90, minibatch_size):
        Xi = Xb_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        grad = Xi.T.dot(Xi.dot(theta)-yi) * (2/minibatch_size)
        theta = theta - lr*grad
        theta_0_path.append(theta[0])
        theta_1_path.append(theta[1])
        theta_2_path.append(theta[2])
        theta_3_path.append(theta[3])
        lr *= wgt_decay # update lr
    
print(theta)
# plotting
plt.plot(X,y,'k.')
plt.plot(xaxis,yaxis,'b--')
plt.plot(xaxis,xaxis**3*theta[0]+xaxis**2*theta[1]+xaxis*theta[2]+theta[3],'r-')
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18) 
plt.show()


# In[94]:


X_new = np.arange(90,108)
X_new_b = np.c_[np.ones((18, 3)), X_new]  # add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# In[95]:


#120 days dataset
xaxis=np.arange(120).reshape((120, -1))
yaxis=covid[:]
plt.plot(xaxis,yaxis,'b.')

#prediction
xx = np.linspace(0,90,100) # for first 90 days
yy = theta[3]*xx**3+theta[2]*xx**2+theta[1]*xx+theta[0]
plt.plot(xx, yy, "r--")

xxx = np.linspace(90,120,100) # for remaining 30 days
yyy = theta[3]*xxx**3+theta[2]*xxx**2+theta[1]*xxx+theta[0]
plt.plot(xxx, yyy, "g--")
plt.title('mini-batch Stochastic Gradient Descent cubic')


# #### 4. Comment on the models in Problems 1/2/3  and select the best model for prediction. 

# In general, the mean square errors from problem 1 to 3 gets lower each time we increased the degree of our polynomial regression. The lowest mean squared error I got was for the full batch gradient descent model, therefore it is the best model for prediction. 
