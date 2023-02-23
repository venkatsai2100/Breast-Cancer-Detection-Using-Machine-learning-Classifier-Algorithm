#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection Using Machine Learning Classifier

# # Import essential libraries 

# In[1]:


# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization


# # Data Load

# In[2]:


#Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# # Data Manupulation

# In[3]:


cancer_dataset


# In[4]:


type(cancer_dataset)


# In[5]:


# keys in dataset
cancer_dataset.keys()


# In[6]:


# featurs of each cells in numeric format
cancer_dataset['data']


# In[7]:


type(cancer_dataset['data'])


# In[8]:


# malignant or benign value
cancer_dataset['target']


# In[9]:


# target value name malignant or benign tumor
cancer_dataset['target_names']


# In[10]:


# description of data
print(cancer_dataset['DESCR'])


# In[11]:


# name of features
print(cancer_dataset['feature_names'])


# In[12]:


# location/path of data file
print(cancer_dataset['filename'])


# ## Create DataFrame

# In[13]:


# create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))


# In[14]:


# DataFrame to CSV file
cancer_df.to_csv('breast_cancer_dataframe.csv')


# In[15]:


# Head of cancer DataFrame
cancer_df.head(6) 


# In[16]:


# Tail of cancer DataFrame
cancer_df.tail(6) 


# In[17]:


# Information of cancer Dataframe
cancer_df.info()


# In[18]:


# Numerical distribution of data
cancer_df.describe() 


# In[19]:


cancer_df.isnull().sum()


# In[56]:


cancer_df.corr()


# In[54]:


# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)


# In[30]:


cancer_df2.corrwith(cancer_df.target).index


# # Split DatFrame in Train and Test

# In[55]:


# input variable
X = cancer_df.drop(['target'], axis = 1) 
X.head(6)


# In[32]:


# output variable
y = cancer_df['target'] 
y.head(6)


# In[33]:


# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)


# In[34]:


X_train


# In[35]:


X_test


# In[36]:


y_train


# In[37]:


y_test


# # Feature scaling 

# In[38]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# # Machine Learning Model Building

# In[39]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ## Suppor vector Classifier

# In[71]:


# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)


# In[72]:


# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)


# # Logistic Regression

# In[80]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# In[79]:


# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_lr_sc)


# # K – Nearest Neighbor Classifier

# In[84]:


# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[86]:


# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)


# # Naive Bayes Classifier

# In[60]:


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)


# In[61]:


# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)


# # Decision Tree Classifier

# In[82]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


#  # Random Forest Classifier

# In[83]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))

