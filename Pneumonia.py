#!/usr/bin/env python
# coding: utf-8

# In[1]:


train_root  = r"C:\Users\CHENMONING\Desktop\2022NTU\CNN\chest_xray\train"
test_root = r"C:\Users\CHENMONING\Desktop\2022NTU\CNN\chest_xray\test"
print(train_root)


# In[2]:


batch_size = 5


# In[3]:


from keras.preprocessing.image import ImageDataGenerator

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (150, 150), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (150, 150), batch_size=batch_size)


# In[4]:


import tensorflow as tf
from matplotlib.pyplot import imshow
import os

im = train_data[0][0][1]
img = tf.keras.preprocessing.image.array_to_img(im)
imshow(img)

num_classes = len([i for i in os.listdir(train_root)])
print(num_classes)


# In[5]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation="softmax"))
model.summary()


# In[6]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_data, batch_size = batch_size, epochs=4)


# In[7]:


score = model.evaluate(test_data)
print(score)


# In[10]:


#remove optimizer if needed
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_data, batch_size = batch_size, epochs=10)

# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)


# In[11]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[12]:


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[16]:


import seaborn as sns
import numpy as np
predict_x=model.predict(test_data) 
pred=np.argmax(predict_x,axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_data.classes, pred)
sns.heatmap(cm, annot=True)


# In[19]:


#depends on number of classes
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[21]:


from tensorflow.keras.models import save_model
save_model(model, "Pneumonia")


# In[5]:


from keras.models import load_model
from PIL import Image #use PIL
import numpy as np

model = load_model("Pneumonia")
import cv2
image = cv2.imread("C:/Users  /CHENMONING/Desktop/2022NTU/CNN/chest_xray/test/PNEUMONIA/person16_virus_47.jpeg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img = cv2.merge([gray,gray,gray])

img.resize((150,150,3))
img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
img.shape
print(model.predict(img))


# In[ ]:




