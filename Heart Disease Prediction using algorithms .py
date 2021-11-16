#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing librabries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
heartdata = pd.read_csv("D:\PERSONAL\LEARNING\ML and Python\DataSet\heartd.csv") 
heartdata.head(100)
#ap_hi-systolic
#ap_low-diastolic


# In[2]:


#counts number of empyt value in each coulmn
heartdata.isnull().sum()


# In[3]:


#view some basic statistic
heartdata.describe()


# In[4]:


#Get the count of number of patients with and without heart disease
# 1 means having disease and 0 means not having 
print(heartdata['target'].value_counts())
sns.countplot(heartdata['target'])


# In[5]:


pd.crosstab(heartdata.age,heartdata.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[6]:


heartdata.corr()


# In[7]:


plt.figure(figsize=(9,9))
sns.heatmap(heartdata.corr(), annot=True, fmt='.0%')


# In[8]:


#splitting data in features and target
Y =heartdata.target.values
x1=heartdata.drop(["target"],axis=1)
# X= heartdata.iloc[:, :-1].values
# Y = heartdata.iloc[:,-1].values


# In[9]:


#Normalization 
X = (x1 - np.min(x1))/(np.max(x1)-np.min(x1)).values


# In[10]:


#Spliting traning and testing data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.2, random_state=37)


# In[11]:


#transposition
xtrain = xtrain.T
xtest = xtest.T
ytrain = ytrain.T
ytest = ytest.T
#defining accuries list to store accuracy of different models
models_test_accuracies = {}
models_train_accuracies = {}


# In[12]:


#LR with sklearn
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain.T,ytrain.T)
train_acc = LR.score(xtrain.T,ytrain.T)*100
models_train_accuracies['LogisticRegression'] = train_acc
print("LogisticRegression Traning Accuracy",train_acc)
print("LogisticRegression Traning Accuracy in % : {:.2f}%".format(train_acc))


# In[13]:


#testing model accuracy on testing data
# arry = [[20228,1,156,85,140,90,3,1,0,0,1]]
print(LR.predict(xtest.T))
test_acc = LR.score(xtest.T,ytest.T)*100
models_test_accuracies['LogisticRegression'] = test_acc
print("LogisticRegression Testing Accuracy  : ",test_acc)
print("LogisticRegression Testing Accuracy in % : {:.1f}%".format(test_acc))
models_test_accuracies


# In[14]:


#Using kneighbour classigier
from sklearn.neighbors import KNeighborsClassifier
kmodel=KNeighborsClassifier(n_neighbors=2)
kmodel.fit(xtrain.T,ytrain.T)
train_acc = kmodel.score(xtrain.T,ytrain.T)*100
models_train_accuracies['KNN'] = train_acc
print("KNN Traning Accuracy",train_acc)
print("KNN Traning Accuracy in  % : {:.2f}%".format(train_acc))


# In[17]:


print(kmodel.predict(xtest.T))
test_acc = kmodel.score(xtest.T,ytest.T)*100
models_test_accuracies['KNN'] = test_acc
print("KNN Test Accuracy ",test_acc)
print("KNN Algorithm Accuracy Score %: {:.2f}%".format(test_acc))


# In[16]:


models_test_accuracies


# In[20]:


#PLOTTING MODELS TRANING ACCURACY
colors = ["yellow", "red"]
sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.yticks(np.arange(0,101,10))
plt.title("Comparision of Algorithms on Training data")

plt.ylabel("Traninig-Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=list(models_train_accuracies.keys()), y=list(models_train_accuracies.values()), palette=colors)
plt.show()


# In[21]:


#PLOTTING MODELS TESTING ACCURACY
colors = ["cyan", "blue"]
sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
plt.yticks(np.arange(0,101,10))
plt.title("Comparision of Algorithms on Testing data")
plt.ylabel("Testing-Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=list(models_test_accuracies.keys()), y=list(models_test_accuracies.values()), palette=colors)
plt.show()


# In[ ]:




