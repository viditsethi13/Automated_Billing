from flask import render_template, url_for, flash, redirect, request
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2


import os
import matplotlib.pyplot as plt
import random as rn
import scipy
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

#from tensorflow.contrib.keras.python.keras.backend import clear_session
from keras.backend import clear_session

posts = [
    {
        'author': 'Shubhanshu Saxena',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'March 30, 2020'
    },
    {
        'author': 'Vidit Sethi',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 15, 2020'
    },
    {
        'author': 'Vipul Somani',
        'title': 'Blog Post 3',
        'content': 'Third post content',
        'date_posted': 'April 23, 2020'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html',posts = posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    return render_template('account.html')


@app.route("/capture",methods=['GET','POST'])
@login_required

def capture():
    
    if request.method=='POST':
        
        cap = cv2.VideoCapture(0)
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
            cv2.imshow('Input', frame)

            c = cv2.waitKey(1)
            if c == 27:
                cv2.imwrite('D:\Education\Automated Billing\Flask\flaskblog\templates\Test.png',frame)
                print("Image Saved")
                cv2.waitKey(0)
                break

        cap.release()
        cv2.destroyAllWindows()
        
        return render_template('capture.html',title='capture')


@app.route("/predict",methods=['GET','POST'])
@login_required

def predict():
    if request.method=='POST':
        clear_session()
        imgsize=150
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
        model.load_weights("D:\Education\Automated Billing\weights.best.hdf5");

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
            14:"Rasbery",
            16:"Walnut",
            13:"Pineapple",
            15:"Strawberry"
        }
        imgsize = 150

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
            fruit=dict[np.argmax(y_pred[i])]

        return render_template('predict.html',title='predict',fruit=fruit)




@app.route("/cost",methods=['GET','POST'])
def cost():
    if request.method=='POST':
        quantity=request.form['quantity']
        fruit=request.form['product']
        cost=10
        if fruit == 'apricot':
            cost=100;
        elif fruit == 'beetroot':
            cost=10;
        elif fruit ==  'blueberry':
            cost=12;

        elif fruit == 'cauliflower':
            cost=1;
            
        elif fruit == 'dates':
            cost=15;
            
        elif fruit ==  'ginger_root':
            cost=100;             
            
        elif fruit ==  'guava':
            cost=100;
            
        elif fruit ==  'kiwi':
            cost=100;   
            
        elif fruit =='lychee':
            cost=100;   
            
        elif fruit == 'orange':
            cost=100;
            
        elif fruit =='papaya':
            cost=100;
            
        elif fruit == 'rasbery':
            cost=100;
            
        elif fruit =='walnut':
            cost=100;
            
        elif fruit =='apple':
            cost=100;
            
        elif fruit =='pineapple':
            cost=100;

        elif fruit =='strawberry':
            cost=100; 
        else:
            cost=10;         

        quantity= int(quantity)
        cost= int(cost)
        final_cost= quantity * cost
        return render_template('cost.html', title='Cost',fruit=fruit,quantity=quantity,amount=final_cost)