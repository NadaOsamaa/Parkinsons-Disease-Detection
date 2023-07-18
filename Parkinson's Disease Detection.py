#!/usr/bin/env python
# coding: utf-8

# ![header.jpg](attachment:header.jpg)

# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# # <span style='background :#e0cebd; color :#000000' > Uploading the dataset </span>   

# In[4]:


data = pd.read_csv('Parkinsson disease.csv')


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# In[7]:


data = data.drop(columns='name')


# ***

# # <span style='background :#e0cebd; color :#000000' > Correlation Matrix </span>   

# In[61]:


plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='copper');


# ***

# # <span style='background :#e0cebd; color :#000000' > Splitting the dataset </span>   

# In[53]:


x = data.loc[:, data.columns != 'status'].values[:, 1:]
y = data.loc[:, 'status'].values

# splitting 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45)


# ***

# # <span style='background :#e0cebd; color :#000000' > Model Fitting & Calculating the Score </span>   

# In[54]:


model = XGBClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
score = accuracy_score(y_test, predict)
print('Accuracy of the model is: %0.4f ' % score)


# ***

# # <span style='background :#e0cebd; color :#000000' > Confusion Matrix </span>   

# In[57]:


cm = confusion_matrix(y_test, predict)
plt.figure(figsize = (8,6))
fg = sns.heatmap(cm, annot=True, cmap='copper')
figure=fg.get_figure()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix');


# ***
