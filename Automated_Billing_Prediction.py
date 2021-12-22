#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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
#from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#from generator import DataGenerator
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[8]:


model=vv()
model.load_weights("D:\Education\Automated Billing\weights.best.hdf5");


# In[9]:


dict = {
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
    14:"Rasbery",
    16:"Walnut",
    1:"Apple",
    13:"Pineapple",
    15:"Strawberry"
}


# In[13]:


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


# In[14]:


path='D:\Education\Automated Billing\Testing'

#testing_data(path)
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


# In[ ]:





# In[ ]:




