#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__
#get_ipython().run_line_magic('load_ext', 'tensorboard')
import xgboost as xgb
xgb.__version__
import numpy as np
import pandas as pd



# In[2]:


import datetime
mnist=tf.keras.datasets.mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train,X_test=X_train/255,X_test/255
X_train=np.reshape(X_train,(60000,784))
#Y_train=np.reshape(X_train,(60000,284))
X_test=np.reshape(X_test,(10000,784))
# In[3]:


train =xgb.DMatrix(X_train,label=Y_train)
test = xgb.DMatrix(X_test,label=Y_test)

param={
    'max_depth':30,
    'eta':0.03,
    'objective':'multi:softmax',
 #   n_estimators 'num_round':20,
    'num_class':10}
epoch=30
model=xgb.train(param,train,epoch)

Y_predict =model.predict(test)
Y_train_predict=model.predict(train)
#print(Y_predict)
#print(Y_train_predict)







from sklearn.metrics import accuracy_score,classification_report

print('Train accuracy',(accuracy_score(Y_train,Y_train_predict)))
print('Test Accuracy',(accuracy_score(Y_test,Y_predict)))
print(classification_report(Y_test,Y_predict))

#pd.crosstab(Y_test,Y_predict)






# In[4]:



# In[5]:





# In[7]:


#get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# import pkg_resources
# 
# for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
#     print(entry_point.dist)

# In[ ]:




