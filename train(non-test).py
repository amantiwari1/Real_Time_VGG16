
# ========================= Import =================================

import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Lambda
import keras
import numpy as np
import glob
from config import n , max1, max2, epochs
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

print("\n\t\t________________________LOADING...______________________\t\t\n")




# ========================= read Images =================================


for layer in vgg16.layers:
  layer.trainable = False

names  = [] 
a = glob.glob("image/*")
b = " ".join(a).split(" ")
for i in b:
	names.append(i.split('\\')[1])


class_names = names

train = []
labels = []
X_test = []
y_test = []




for num,name in enumerate(names):
	for i in range(1,max1):
	    a = cv2.imread("image/"+name+"/"+name+".{}.jpg".format(i))
	    labels.append(num)
	    train.append(a)
    

for num,name in enumerate(names):    
	for i in range(1,max2):
	    a = cv2.imread("image_test/"+name+"/"+name+".{}.jpg".format(i))
	    y_test.append(num)
	    X_test.append(a)

    
X_test = np.array(X_test)
y_test = np.array(y_test)
train = np.array(train)
labels = np.array(labels)

X_test= X_test/255.0
train = train/255.0

print(labels[:5])
print(labels[193:199])




# ========================= Model =================================


model = Sequential()

model.add(Conv2D(128, (3,3), activation="relu", input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(n, activation="softmax"))

model.compile(optimizer="adam", loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics=["acc"])



# ========================= Training =================================


model.fit(train, labels, epochs=epochs, validation_data = (X_test, y_test))



# ========================= Saved Model =================================

model.save("model.h5")


# ========================= Test Evaluate Accuracy =================================


score  = model.evaluate(X_test, y_test)

print("Total Accuracy  : ", score[1])
print("Successfully......")
print("Please Run python Real_Time.py")




