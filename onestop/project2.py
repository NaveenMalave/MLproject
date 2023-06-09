#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Linear Discriminant Analysis


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


dataset= pd.read_csv("data1.csv")


# In[4]:


dataset


# In[5]:


x = dataset.iloc[:,:-1]
x


# In[6]:


y = dataset.iloc[:,:-1]


# In[7]:


y


# In[19]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


X_train


# In[22]:


y_train


# In[23]:


X_test


# In[24]:


y_test


# In[25]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[26]:


X_train


# In[27]:


X_test


# In[28]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda =LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)


# In[29]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[30]:


X_test


# In[31]:


y_pred


# In[32]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[42]:


from matplotlib.colors import ListedColormap
x_set,y_set = X_train,y_train
x1,x2=np.meshgrid(
   np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.25),
   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.25),
)
plt.contourf(x1,x2,lr.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap=ListedColormap(('red','blue','green')))


# In[43]:


from matplotlib.colors import ListedColormap
x_set,y_set = X_train,y_train
x1,x2=np.meshgrid(
   np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.25),
   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.25),
)
plt.contourf(x1,x2,lr.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap=ListedColormap(('red','blue','green')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x1.min(),x1.max())


# In[45]:


from matplotlib.colors import ListedColormap
x_set,y_set = X_train,y_train
x1,x2=np.meshgrid(
   np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.25),
   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.25),
)
plt.contourf(x1,x2,lr.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap=ListedColormap(('red','blue','green')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x1.min(),x1.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],
               c= ListedColormap(('red','blue','green'))(i),label = j )
plt.title("Test Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




