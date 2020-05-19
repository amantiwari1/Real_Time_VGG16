#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
from keras.layers import Dropout,Dense, Conv2D, MaxPooling2D, Flatten, Input, Lambda
import keras
import numpy as np
import glob
#from config import n , max1, max2, epochs
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


# In[2]:



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[3]:


training_set = train_datagen.flow_from_directory('image',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[4]:



test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('image_test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[6]:


model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224, 224, 3))


# In[7]:


def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model


# In[14]:


FC_Head = addTopModel(model, 2)

modelnew = Model(inputs=model.input, outputs=FC_Head)
modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

modelnew.summary()


# In[19]:


r = modelnew.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
)
modelnew.save("test_vgg.h5")


# In[ ]:




