
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
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
#from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
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


apricot = 'E:\Automated Billing\Fruit DataSet\Images\APRICOT'
avocado = 'E:\Automated Billing\Fruit DataSet\Images\AVOCADO'
beetroot= 'E:\Automated Billing\Fruit DataSet\Images\BEETROOT'
blueberry = 'E:\Automated Billing\Fruit DataSet\Images\BLUEBERRY'
cauliflower = 'E:\Automated Billing\Fruit DataSet\Images\CAULIFLOWER'
dates = 'E:\Automated Billing\Fruit DataSet\Images\DATES'
ginger_root = 'E:\Automated Billing\Fruit DataSet\Images\GINGER_ROOT'
guava = 'E:\Automated Billing\Fruit DataSet\Images\GUAVA'
kiwi= 'E:\Automated Billing\Fruit DataSet\Images\KIWI'
lychee = 'E:\Automated Billing\Fruit DataSet\Images\LYCHEE'
orange = 'E:\Automated Billing\Fruit DataSet\Images\ORANGE'
papaya = 'E:\Automated Billing\Fruit DataSet\Images\PAPAYA'
rasbery = 'E:\Automated Billing\Fruit DataSet\Images\RASPBERY'
walnut = 'E:\Automated Billing\Fruit DataSet\Images\WALNUT'
apple = 'E:\Automated Billing\Fruit DataSet\Images\APPLE'
pineapple = 'E:\Automated Billing\Fruit DataSet\Images\PINEAPPLE'
strawberry = 'E:\Automated Billing\Fruit DataSet\Images\STRAWBERRY'

X = []
Z = []
imgsize = 150


# Function returning Model

# In[7]:


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


model=vv()
model.load_weights("E:\Automated Billing\weights.best.hdf5");


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
        cv2.imwrite('E:\Automated Billing\Testing\messigray.png',frame)
        print("Image Saved")
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()


# In[22]:


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

# In[23]:


path='E:\Automated Billing\Testing'

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






