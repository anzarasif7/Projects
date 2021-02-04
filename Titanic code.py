#!/usr/bin/env python
# coding: utf-8

# In[26]:



import pandas as pd
from matplotlib import pyplot as plt


# In[27]:


titanic_train=pd.read_csv('train.csv')
titanic_test=pd.read_csv('test.csv')


# In[28]:


titanic_train.head()


# In[29]:


titanic_train.shape


# In[30]:


titanic_train['Survived'].value_counts()


# In[31]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color=["r","g"])
plt.show()


# In[32]:


titanic_train['Pclass'].value_counts()


# In[33]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Pclass'].value_counts().keys()),list(titanic_train['Pclass'].value_counts()),color=["orange"])
plt.show()


# In[34]:


titanic_train['Sex'].value_counts()


# In[35]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color=["blue","pink"])
plt.show()


# In[36]:


plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.show()


# In[37]:


titanic_train['Survived'].isnull()


# In[38]:


sum(titanic_train['Survived'].isnull())


# In[39]:


titanic_train['Age'].isnull()


# In[40]:


sum(titanic_train['Age'].isnull())


# In[41]:


titanic_train=titanic_train.dropna()


# In[42]:


x_train=titanic_train[['Age']]
y_train=titanic_train[['Survived']]


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


dtc= DecisionTreeClassifier()


# In[45]:


dtc.fit(x_train,y_train)


# In[46]:


#predicting values


# In[47]:


sum(titanic_test['Age'].isnull())


# In[48]:


titanic_test=titanic_test.dropna()


# In[49]:


x_test=titanic_test[['Age']]


# In[50]:


y_pred=dtc.predict(x_test)


# In[51]:


y_pred


# In[ ]:





# In[ ]:




