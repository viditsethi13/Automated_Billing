#!/usr/bin/env python
# coding: utf-8

# In[25]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("D:\Education\Automated Billing" ):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import scipy
import os
import shutil
import math
import scipy
import cv2

from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam,SGD

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# from generator import DataGenerator
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler



# In[28]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


# In[29]:


def label_assignment(img,label):
    return label

def training_data(label,data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img,label)
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        
        X.append(np.array(img))
        Z.append(str(label))


# In[30]:


apricot = 'D:\Education\Automated Billing\Fruit DataSet\Images\APRICOT'
avocado = 'D:\Education\Automated Billing\Fruit DataSet\Images\AVOCADO'
beetroot= 'D:\Education\Automated Billing\Fruit DataSet\Images\BEETROOT'
blueberry = 'D:\Education\Automated Billing\Fruit DataSet\Images\BLUEBERRY'
cauliflower = 'D:\Education\Automated Billing\Fruit DataSet\Images\CAULIFLOWER'
dates = 'D:\Education\Automated Billing\Fruit DataSet\Images\DATES'
ginger_root = 'D:\Education\Automated Billing\Fruit DataSet\Images\GINGER_ROOT'
guava = 'D:\Education\Automated Billing\Fruit DataSet\Images\GUAVA'
kiwi= 'D:\Education\Automated Billing\Fruit DataSet\Images\KIWI'
lychee = 'D:\Education\Automated Billing\Fruit DataSet\Images\LYCHEE'
orange = 'D:\Education\Automated Billing\Fruit DataSet\Images\ORANGE'
papaya = 'D:\Education\Automated Billing\Fruit DataSet\Images\PAPAYA'
rasbery = 'D:\Education\Automated Billing\Fruit DataSet\Images\RASPBERY'
walnut = 'D:\Education\Automated Billing\Fruit DataSet\Images\WALNUT'
apple = 'D:\Education\Automated Billing\Fruit DataSet\Images\APPLE'
pineapple = 'D:\Education\Automated Billing\Fruit DataSet\Images\PINEAPPLE'
strawberry = 'D:\Education\Automated Billing\Fruit DataSet\Images\STRAWBERRY'

X = []
Z = []
imgsize = 150

training_data('apricot', apricot)
training_data('avocado',avocado)
training_data('beetroot',beetroot)
training_data('blueberry', blueberry)
training_data('cauliflower',cauliflower)
training_data('dates',dates)
training_data('ginger_root', ginger_root)
training_data('guava',guava)
training_data('kiwi', kiwi)
training_data('lychee',lychee)
training_data('apple',apple)
training_data('pineapple',pineapple)
training_data('strawberry',strawberry)
training_data('orange', orange)
training_data('papaya',papaya)
training_data('rasbery',rasbery)
training_data('walnut',walnut)


# Spliting Data: Train and Test

# In[31]:


label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y,17)
X = np.array(X)
X=X/255

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))


# Image Rotation and Flip

# In[32]:


augs_gen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        horizontal_flip=True,  
        vertical_flip=False) 

augs_gen.fit(x_train)


# Function returning Model

# In[33]:


def vv():
    base_model = VGG16(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = 'imagenet')

    for layer in base_model.layers:
        layer.trainable = False
    
    for layer in base_model.layers:
        print(layer,layer.trainable)

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(17,activation='softmax'))
    return model
imgsize = 150


# Weights

# In[34]:


from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


# Training


# In[35]:


model = vv()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(
    augs_gen.flow(x_train,y_train,batch_size=25),
    validation_data  = (x_test,y_test),
    validation_steps = 100,
    steps_per_epoch  = 100,
    epochs = 50, 
    verbose = 1,
    callbacks=callbacks_list
)


# Loading Weights into Model

# In[36]:


model=vv()
model.load_weights("D:\Education\Automated Billing\weights.best.hdf5");


# Label: Configration

# In[37]:


dict = {
    0:"Apple",
    1:"Apricot",
    2:"Avocado",
    3:"Beetroot",
    4:"Blueberry",
    5:"Cauliflower",
    6:"Dates",
    7:"Ginger_root",
    8:"Guava",
    9:"Kiwi",
    10:"Lychee",
    11:"Orange",
    12:"Papaya",
    13:"Pineapple",
    14:"Rasbery",
    15:"Strawberry",
    16:"Walnut"    
}


# Starting WebCam and Saving Image in Testing Folder

# In[38]:


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        cv2.imwrite('D:\Education\Automated Billing\messigray.png',frame)
        print("Image Saved")
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()


# In[39]:


imgsize = 150

def testing_data(data_dir):
    TT = []
    for img in tqdm(os.listdir(data_dir)):
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        TT.append(np.array(img))
        TT = np.array(TT)
        TT=TT/255


# Prediction Function

# In[40]:


path='D:\Education\Automated Billing\Testing'

TT = []
for img in tqdm(os.listdir(path)):
    path = os.path.join(path,img)
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(imgsize,imgsize))
    TT.append(np.array(img))
    TT = np.array(TT)
    TT=TT/255
y_pred = model.predict(TT)
for i in range(len(y_pred)):
    print(dict[np.argmax(y_pred[i])])


# Model Compile and Evaluate

# In[41]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
loss,acc = model.evaluate(x_test,y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:




