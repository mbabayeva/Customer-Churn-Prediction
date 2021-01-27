#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Perceptron :
# 
# ## It's one of the simplest ANN model, which is slightly different artificial neuron called linear threshold unit. 
#  
# ![image.png](attachment:image.png)
# 
# 
# 
# ### A single perceptron can only be used to implement linearly separable functions. It takes both real and boolean inputs and associates a set of weights to them, along with a bias.
# 
# ## Update Rule :
#                                                      w(i,j)(new) = w(i,j)(old) + n (y_hat(j) - y(j))x(i)
# 
#          w(i,j)   ---> connection weight between i th input neuron and j th output neuron.
#          x(i)     ---> is the i th input value of the current training instance.
#          y_hat(j) ---> is the output of j th output neuron for the current training instance.
#          y(j)     ---> is the j th target output of the j th output neuron for the current training instance.
#          n        ---> is the learning rate.
# 

# This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.
# Nowadays, the telecommunication industries are facing substantial competition among the providers in order to capture new customers. Many providers have faced a loss of profitability due to the existing customers migrating to other providers. Customer retention program is one of the main strategies adopted in order to keep customers loyal to their provider. However, it requires a high cost and therefore the best strategy that companies could practice is to focus on identifying the customers that have the potential to churn at an early stage. The limited amount of research on investigating customer churn using machine learning techniques has lead this research to explore the potential of an artificial neural network to improve customer churn prediction.

# In[2]:


data = pd.read_csv('Churn_Modelling.csv')


# ## EDA

# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.head()


# In[6]:


data['Geography'].value_counts()


# In[7]:


data['EstimatedSalary'] = data['EstimatedSalary'].astype(int) ## EstimatedSalary col was in float converted into int datatype 


# In[8]:


data.drop(columns=['RowNumber', 'CustomerId', 'Surname','Geography'],inplace=True) ## no impact in the output


# In[9]:


data.head()


# ## convert categorical variables into numeric form.

# In[10]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[11]:


data['Gender'] = le.fit_transform(data['Gender'])


# In[12]:


data


# In[39]:


X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values


# ## Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias. Thus, feature-wise normalization such as MinMax Scaling is usually used.

# In[40]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[41]:


X = scaler.fit_transform(X)


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[43]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Perceptron Model Implementation

# In[44]:


from sklearn.linear_model import Perceptron
clf = Perceptron()


# In[45]:


clf.fit(X_train,y_train)


# In[46]:


y_pred = clf.predict(X_test)


# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


accuracy_score(y_test,y_pred)


# ## Hyperparameter tuning using gridsearchcv

# In[49]:


param_dist = {
    'penalty' :['l2','l1','elasticnet',None],
    'alpha':[0.001,0.0001,0.00001],
    'max_iter':[10,100,1000,10000]
    
}


# In[50]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf,param_grid=param_dist,cv=10)


# In[51]:


grid.fit(X_train,y_train)


# ## Best Estimator

# In[52]:


grid.best_estimator_


# ## Best Score

# In[53]:


grid.best_score_


# In[54]:


updated_clf = Perceptron(alpha=0.001, max_iter=10, penalty='l1')


# In[55]:


updated_clf.fit(X_train,y_train)


# In[56]:


updated_y_pred = updated_clf.predict(X_test)


# ## Accuracy

# In[57]:


accuracy_score(y_test,updated_y_pred)


# In[ ]:




