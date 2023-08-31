#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Kyphosis_df = pd.read_csv("kyphosis.csv")


# In[3]:


Kyphosis_df.head(10)


# In[4]:


Kyphosis_df.tail()


# In[5]:


Kyphosis_df.describe()


# In[6]:


Kyphosis_df.info()


# In[7]:


sns.countplot(Kyphosis_df['Kyphosis'], label = "Count") 


# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_y = LabelEncoder()
Kyphosis_df['Kyphosis'] = LabelEncoder_y.fit_transform(Kyphosis_df['Kyphosis'])


# In[9]:


Kyphosis_df


# In[10]:


Kyphosis_True = Kyphosis_df[Kyphosis_df['Kyphosis']==1]


# In[11]:


Kyphosis_False = Kyphosis_df[Kyphosis_df['Kyphosis']==0]


# In[12]:


print( 'Disease present after operation percentage =', (len(Kyphosis_True) / len(Kyphosis_df) )*100,"%")


# In[13]:


plt.figure(figsize=(10,10)) 
sns.heatmap(Kyphosis_df.corr(), annot=True) 


# In[14]:


sns.pairplot(Kyphosis_df, hue='Kyphosis', vars = ['Age', 'Number', 'Start'])


# In[15]:


# Let's drop the target label coloumns
X = Kyphosis_df.drop(['Kyphosis'],axis=1)
y = Kyphosis_df['Kyphosis']


# In[16]:


X


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[20]:


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# In[21]:


X_train.shape


# In[22]:


y_train.shape


# In[23]:


X_test.shape


# In[24]:


y_test.shape


# In[25]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)


# In[26]:


feature_importances = pd.DataFrame(decision_tree.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[27]:


feature_importances


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix


# In[29]:


y_predict_train = decision_tree.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[30]:


# Predicting the Test set results
y_predict_test = decision_tree.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[31]:


print(classification_report(y_test, y_predict_test))


# In[32]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators=150)
RandomForest.fit(X_train, y_train)


# In[33]:


y_predict_train = RandomForest.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[34]:


# Predicting the Test set results
y_predict_test = RandomForest.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[35]:


print(classification_report(y_test, y_predict_test))

