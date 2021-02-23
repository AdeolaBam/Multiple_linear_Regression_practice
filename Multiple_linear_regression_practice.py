#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("50_Startups.csv")


# In[15]:


X=df.iloc[:, :-1]


# In[16]:


X.head()


# In[17]:


Y=df.iloc[:,4]


# In[18]:


Y.head()


# In[20]:


df["State"].unique()


# In[34]:


df["State_new"]=df["State"].sort_index()


# In[36]:


df.head()


# In[43]:


df.iloc[:,5].unique()


# In[44]:


org_labels=df.iloc[:,5].unique()


# In[45]:


enumerate(org_labels,0)


# In[47]:


org_labels2={k:i for i,k in enumerate(org_labels,0)}
org_labels2


# In[49]:


df['State_new']=df['State'].map(org_labels2)
df.head()


# In[21]:


#converting categorical variables colum 
df['State']=df['State'].astype(str).str[0]


# In[50]:


df.drop('State',axis=1,inplace=True)


# In[51]:


df.head()


# In[54]:


#df.iloc[0,1,2,4]
df.iloc[:, lambda df: [0, 1,2,4]].head()


# In[55]:


X=df.iloc[:, lambda df: [0, 1,2,4]].head()


# In[56]:


X.head()


# In[58]:


Y=df.iloc[:,3]
Y.head()


# In[66]:


X


# In[67]:


df


# In[68]:


X=df.iloc[:, lambda df: [0, 1,2,4]]


# In[69]:


X


# In[70]:


Y=df.iloc[:,3]


# In[71]:


Y


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)


# In[74]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[75]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[78]:


y_pred


# In[79]:


y_test


# In[80]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


# In[81]:


score


# In[ ]:





# In[ ]:



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


# In[ ]:





# In[ ]:




