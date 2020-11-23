#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
print(os.listdir())     #Will tell Files in Working Directory

# Importing Data

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[:,1:].values
y_train = train_data.iloc[:,0].values

#Checking For Any Null Values
pd.DataFrame(X_train).isna().sum()
test_data.isna().sum()

sns.countplot(y_train)  #Graphically checking how many datasets we have

#Scaling and Reshaping Train test data 
X_train = X_train/255.0
test_data = test_data/255.0


X_train = X_train.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes=10)   

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.1,random_state=2)

X_train2 = 1 - X_train
X_test2 = 1 - X_test

X_train = np.concatenate((X_train,X_train2),axis = 0)
y_train = np.concatenate((y_train,y_train),axis=0)
X_test = np.concatenate((X_test,X_test2),axis = 0)
y_test =  np.concatenate((y_test,y_test),axis = 0)
#g = plt.imshow(X_train[5][:,:,0])


#Preparing Our CNN Model


from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D+++


















from keras.layers import MaxPool2D
from keras.layers import Flatten

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

classifier.fit(X_train,y_train,batch_size = 200,epochs=10,validation_data=(X_test,y_test)) 
 

#Checking results on test_data
result = classifier.predict(test_data)        

results = np.argmax(result,axis = 1)     #Predictions of test_data

#Working On External Images
from PIL import Image
im = Image.open('handwritten5.jpg')
im = im.resize((28,28))
im.save('updated.png')

#Single Digit
import cv2
im = cv2.imread('updated.png',0)/255.0
for i in range(28):
    for j in range(28):
        im[i][j] = round(im[i][j],0)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()


OP = classifier.predict(im.reshape(-1,28,28,1))        
op = np.argmax(OP,axis = 1)




#Phone No

from PIL import Image
im = Image.open('nos.jpg')
im = im.resize((560,56))
im.save('updated.png')


im = cv2.imread('updated.png',0)/255.0
for i in range(56):
    for j in range(560):
        im[i][j] = round(im[i][j],0)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
list_im = []
imlist = []


import numpy as np

for j in range(560):
    if np.all(im[:,j] == 1):
        list_im.append(j)

          
if 0 in list_im:  
    list_im.remove(0)  
    list_im.insert(0,1)
    
if 559 in list_im:
    list_im.remove(559)   
      

k=140
while k:
    for j in list_im:
        if np.all(im[:,j-1] == 1) and np.all(im[:,j+1]==1):
            list_im.remove(j)
    k=k-1

imlist = [0,]


for i in range(len(list_im)):
    if i==19:
        break

    if i%2 != 0:
        imlist.append(int((list_im[i] + list_im[i+1])/2))

list_im = []
imlist.append(559)

for j in range(len(imlist)):
    if j==10:
        break        
    img = im[:,imlist[j]:imlist[j+1]]
    list_im.append(img)





PhoneNo = []



for i in range(len(list_im)):              
    img = Image.fromarray(list_im[i])   
    img = img.resize((28,28))


    img = np.array(img)
    img = img.reshape(-1,28,28,1)


    OP = classifier.predict(img)        
    op = np.argmax(OP,axis = 1)    

    PhoneNo.append(op)



im = train_data.iloc[6,1:].values.reshape(28,28)/255.0

plt.imshow(img.reshape(28,28))

plt.imshow(img)
