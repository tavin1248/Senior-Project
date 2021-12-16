""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

# This file contains all of the views for learning website. 
# This file also contains all of the backend Neural Network programming and data assignment

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from . models import Post, Images, GoodTrainingImages, BadTrainingImages, GoodValidationImages, BadValidationImages, GoodTestImages, BadTestImages, LoadModel
from django.views.generic import (ListView,
                                  DetailView,
                                  CreateView,
                                  UpdateView,
                                  DeleteView,
                                  FormView,
                                  )
from .forms import *
from django.contrib import messages
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
from keras import models
from keras import layers
from tensorflow.keras import optimizers
from keras import models
from keras import layers
from django.urls import reverse
from django.urls import path
import pandas
from PIL import Image
from skimage import transform
import mimetypes
from django.http import HttpResponse
import glob
keras.__version__

# this function represents the main view of the program
def home(request):
    # printing all of the posts by the user
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'learningWebsite/home.html', context)

# this class allows each post to be order by date in a list and also assigns a html template
class PostListView(ListView):
    model = Post
    template_name = 'learningWebsite/home.html'
    context_object_name = 'posts'
    ordering = ['-date_created']

# This class defines that only a user can view a users posts
# This class also assigns a pagination value so that the webpages do not scroll,
# instead the webpages have 'pages' of posts
class UserPostListView(ListView):
    model = Post
    template_name = 'learningWebsite/user_posts.html'
    context_object_name = 'posts'
    paginate_by = 6
 
    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_created')
 
# This class allows a user to enter into a specific project and look at the details within that project.
class PostDetailView(DetailView):
    context_object_name = 'learningWebsite/post_detail.html'
    template_name = 'learningWebsite/post_detail.html'
    queryset = Post.objects.all()   # printing all of the information within the project class for this object

# this class allows a user to create a new project and assign it a title
class PostCreateView(LoginRequiredMixin, CreateView):

    model = Post

    fields = ['title']
    
    # function to make sure only a user that is signed in can create a project
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

# This class allows a user to update a project
class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title']
    
    # functions to verify that only the user who created the project can edit the project
    def from_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

#This function allows users to delete projects
class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'

    # verifying that the author can only delete the post
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
 
# This is the view for a user who is currently testing a new deep learning model
def testing(request):
    return render(request, 'learningWebsite/testing.html',{'title': 'training'})

# this function allows a user to view the output of their deep learning model
def output(request):
    return render(request, 'learningWebsite/output.html',{'title': 'output'})
 
# This function simply lets the user know that their deep learning model training is complete
def post_training(request):
 
    context = {
        'trainings': Training.objects.all()
    }
    return render(request, 'learningWebsite/training.html', context)

# This function lets the user know that their model is currently training 
def TrainLoad(request):

    return render(request, 'learningWebsite/train_load.html')

# This is the main training function for the deep learning model
# This function uses CNN + DA in order to train the model. 
# After the model is trained, the model is saved
def TrainingAlgorithm(request):

    #Defining the directories within the amazon S3 bucket
    train_dir = 'MachineLearning/Training/'
    vali_dir = 'MachineLearning/Validation/'
    test_dir = 'MachineLearning/Test/'

    train_bad_dir = 'MachineLearning/Train/bad_training_images/'
    train_good_dir = 'MachineLearning/Train/good_training_images/'
    vali_bad_dir = 'MachineLearning/Validation/bad_validation_images/'
    vali_good_dir = 'MachineLearning/Validation/good_validation_images/'
    test_bad_dir = 'MachineLearning/Test/bad_test_images/'
    test_good_dir = 'MachineLearning/Test/good_test_images/'

    goodPartNumber = 0
    badPartNumber = 0
    goodValNumber = 0
    badValNumber = 0
    goodTestNumber = 0
    badTestNumber = 0

    # in case the number of parts within the directories need to be printed to the user
    context = {
        'goodPartNumber': len(train_good_dir),
        'badPartNumber': len(train_bad_dir),
        'goodValNumber': len(vali_good_dir),
        'badValNumber': len(vali_bad_dir),
        'goodTestNumber': len(test_good_dir),
        'badTestNumber': len(test_bad_dir),
    }

    # the build model function allows the model to define what type of training it will use
    # for this application, I chose a CNN + DA due to a limited number of parts and necessity
    # to augment the inputted data for increased reliability when testing
    def build_model():

        model = models.Sequential()
        model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape = (150,150, 3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())                            
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=1e-4),
                    metrics=['acc'])
        return model

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    train_generator = train_datagen.flow_from_directory(
            train_dir,              
            target_size=(150, 150),
            batch_size=10,               
            class_mode='binary')    
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
            vali_dir,
            target_size=(150, 150),
            batch_size=10,
            class_mode='binary')

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    model = build_model()

    # A better method might be to use 'glob.glob', iterate through the files in the directory
    # with a count variable, then assign the count variable to the necessary number of epoch
    # within the steps and validation_steps so that inputted data is variable.
    history = model.fit(
        train_generator,
        steps_per_epoch=79//10,     # 20 (batch_size) x 92 = 1840 train samples
        epochs=200,
        validation_data=validation_generator,
        validation_steps=22//10      # 20 (batch_size) x 26 = 520 validation samples
    )

    # saving the model to a directory for future use
    model.save('MachineLearning/TrainingAlgorithm/Model1.h5')

    # once the training is finished, the page will redirect a user to an evaluation page
    return render(request, 'learningWebsite/Training.html', context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the goodTrainingImage model
def goodTrainingImageView(request):
    
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = GoodTrainingImages.objects.create(image=image,)
            photo.save()
            queryset = GoodTrainingImages.objects.all()
            redirect('good-training-image')
    
    context = {
        'images': GoodTrainingImages.objects.all()
    } 
    return render(request, 'learningWebsite/post_good_training_image.html', context)
    

# this function was meant to allow the user to delete the images within the goodTrainingImageView model
# however, I was unable to get this function to work reliably
class goodTrainingImageViewDelete(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = GoodTrainingImages
    success_url = '/'

    def test_func(self):
        return True

# This function allows the user to download their most recently trained model.
def download_file(request):
    fl_path = 'MachineLearning/TrainingAlgorithm/'
    filename = 'Model1.h5'
    fl = keras.models.load_model('MachineLearning/TrainingAlgorithm/Model1.h5')
    mime_type, _ = mimetypes.guess_type(fl_path)
    response = HttpResponse(fl, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

# This function allows the model to be tested with user entered test data that is pulled from the amazon s3 bucket
def EvaluateModel(request):
    test_dir = 'MachineLearning/Test/'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (150,150),
        batch_size = 22,
        class_mode = 'binary'
    )

    #using loss and accuracy metrics that will be printed to the user
    model = keras.models.load_model('MachineLearning/TrainingAlgorithm/Model1.h5')
    loss, acc = model.evaluate(test_generator, steps = 1)
    acc = acc * 100
    loss = loss * 100

    # rounding to two decimals as to keep the look of the webpage 'clean'
    acc = round(acc, 2)
    loss = round(loss, 2)
    
    context = {
        'acc': acc,
        'loss': loss
    }

    return render(request, 'learningWebsite/Evaluation.html', context)

# This function allows a user entered deep learning model to be saved to the LoadCam model
def LoadCamView(request):
    if request.method == "POST":
        form = LoadCamForm(request.POST, request.FILES)
        #printing a form to allows the user to see what they need to input
        if form.is_valid():
            form.save()
            return redirect('load-model')
    
    else:
        form = LoadCamForm()
    return render(request, 'learningWebsite/LoadCam.html', {'form':form})

# This function allows the inputted camera image to be processed by the deep learning model that was trained
def LoadModelView(request):
        
        # function to resize the inputted image to the appropriate size
        def load(img):
            np_image = Image.open(img)
            np_image = np.array(np_image).astype('float32')/255
            np_image = transform.resize(np_image,(150,150,3))
            np_image = np.expand_dims(np_image, axis=0)
            return np_image

        #where the inputted camera image and model are being pulled from in the amazon s3 bucket
        test_dir = 'MachineLearning/TrainingAlgorithm/Model1.h5'
        img = 'MachineLearning/UserTest/Image/image001.jpg'
        photo = LoadModel.objects.create(image=img,)
        photo.save()
        img = load(img)

        #loading the model and predicting the image with the loaded model
        model = keras.models.load_model(test_dir)
        guess = model.predict(img)

        #calculating whether the inputted image is a good or a bad part
        guess = guess * 100
        guess = np.round(guess, 2)
        guess = guess[0][0]
        partType = ''
        if guess > 75:
            partType = 'Part has no defects'
        else:
            partType = 'Part has defects detected'

        # printing the guess of the DL model, the part type, and the image associated with that guess
        context = {'guess': guess, 'label': partType, 'images': LoadModel.objects.all()[:1]}
        return render(request, 'learningWebsite/LoadModel.html', context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the badTrainingImage model
def badTrainingImageView(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = BadTrainingImages.objects.create(image=image,)
            photo.save()
            redirect('bad-training-image')
    context = {
        'images': BadTrainingImages.objects.all()
    }
    return render(request, 'learningWebsite/post_bad_training_image.html', context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the goodValidationImage model
def goodValidationImageView(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = GoodValidationImages.objects.create(image=image,)
            photo.save()
            redirect('good-validation-image')
    context = {
        'images': GoodValidationImages.objects.all()
    }
    return render(request, 'learningWebsite/post_good_validation_image.html',context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the badValidationImage model
def badValidationImageView(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = BadValidationImages.objects.create(image=image,)
            photo.save()
            redirect('bad-validation-image')
    context = {
        'images': BadValidationImages.objects.all()
    }

    return render(request, 'learningWebsite/post_bad_validation_image.html',context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the goodTestImage model
def goodTestImageView(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = GoodTestImages.objects.create(image=image,)
            photo.save()
            redirect('good-test-image')
    context = {
        'images': GoodTestImages.objects.all()
    }

    return render(request, 'learningWebsite/post_good_test_image.html', context)

# this function allows the user to enter in multiple images at a time,
# iterating through each file and assigning it to the badTestImage model
def badTestImageView(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        for image in images:
            photo = BadTestImages.objects.create(image=image,)
            photo.save()
            redirect('bad-test-image')
    context = {
        'images': BadTestImages.objects.all()
    }

    return render(request, 'learningWebsite/post_bad_test_image.html', context)

#This function simply sends the user to an explanation of what the steps of the website are
def HowToView(request):
    return render(request, 'learningWebsite/HowTo.html')