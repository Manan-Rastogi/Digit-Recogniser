#Importing Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

Image_index = random.randint(0,60000)

plt.imshow(x_train[Image_index])




x_train2 = 1 - x_train
x_test2 = 1 - x_test

x_train = np.concatenate((x_train,x_train2),axis = 0)
y_train = np.concatenate((y_train,y_train),axis=0)
x_test = np.concatenate((x_test,x_test2),axis = 0)
y_test =  np.concatenate((y_test,y_test),axis = 0)





# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)




# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float')/255.0
x_test = x_test.astype('float')/255.0



# Normalizing the RGB codes by dividing it to the max RGB value.

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes=10) 
y_test = to_categorical(y_test,num_classes=10) 


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D


classifier = Sequential()

classifier.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu',padding='same'))

classifier.add(MaxPool2D((2,2)))

classifier.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu',padding='same'))

classifier.add(MaxPool2D(2,2))

classifier.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu',padding='same'))

classifier.add(MaxPool2D(2,2))

classifier.add(Flatten())

classifier.add(Dense(units = 64,activation='relu'))
classifier.add(Dense(units = 10,activation='softmax'))


classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size = 200,epochs=10,validation_data=(x_test,y_test))


classifier.evaluate(x_test, y_test)

Image_index = random.randint(0,20000)

pred = classifier.predict(x_test[Image_index].reshape(1, 28, 28, 1))

pred.argmax()

plt.imshow(x_test[Image_index].reshape(28,28))



#Digit Recogniser
from PIL import Image
im = Image.open('handwritten5.jpg')
im = im.resize((28,28))
im.save('updated.png')


import cv2
im = cv2.imread('updated.png',0)/255.0
for i in range(28):
    for j in range(28):
        im[i][j] = round(im[i][j],0)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

pred = classifier.predict(im.reshape(-1, 28, 28, 1))

plt.imshow(im.reshape(28,28))

pred.argmax()

