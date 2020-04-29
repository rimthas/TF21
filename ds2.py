#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__
#get_ipython().run_line_magic('load_ext', 'tensorboard')
import xgboost as xg
xg.__version__
import numpy as np


# In[2]:


import datetime
mnist=tf.keras.datasets.mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train,X_test=X_train/255,X_test/255


# In[3]:


model =tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                   tf.keras.layers.Dense(1024,activation='relu'),
                                   tf.keras.layers.Dropout(0.1),
                                   tf.keras.layers.Dense(10,'softmax')])
                                   
model.summary()


# In[4]:


model.compile(optimizer="adamax",
             loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[5]:


path="logs\\fit\\"+datetime.datetime.now().strftime("%y%m%d-%H%M%S")
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=path,histogram_freq=1)


# In[6]:
tstart= datetime.datetime.now()

model.fit(x=X_train,
         y=Y_train,
         epochs=20,
         batch_size=64,

         validation_data=(X_test,Y_test),
         callbacks=[tensorboard_callback])

tend = datetime.datetime.now()

timediff=tend-tstart
print(timediff)
import os
os.system('tensorboard --logdir='+path)



# In[7]:


#get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# import pkg_resources
# 
# for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
#     print(entry_point.dist)

# In[ ]:




