#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Hierachical clustering


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[18]:


dataset= pd.read_csv("segmented_customers.csv")
dataset.head()


# In[19]:


dataset.isnull().sum()


# In[20]:


dataset.describe()


# In[3]:


dataset= pd.read_csv("segmented_customers.csv")
x = dataset.iloc[:,:].values


# In[4]:


x


# In[5]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))


# In[6]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("customers")
plt.ylabel("Euclidian distance")
plt.show()


# In[9]:


get_ipython().system('pip install scikit-learn')


# In[12]:


from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=5)
y_hc=clustering.fit_predict(x)


# In[13]:


y_hc


# In[14]:


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c='green',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c='pink',label='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c='blue',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c='orange',label='cluster 5')


# In[15]:


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c='red',label='cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c='green',label='cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c='pink',label='cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c='blue',label='cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c='orange',label='cluster 5')
plt.title("segmented customers")
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.legend()
plt.show()


# In[16]:


x[y_hc==2,1]


# In[21]:


dataset.to_csv("segmented_customers.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:




