""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

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

 
def home(request):
 
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'learningWebsite/home.html', context)
 
class PostListView(ListView):
    model = Post
    template_name = 'learningWebsite/home.html'
    context_object_name = 'posts'
    ordering = ['-date_created']
 
class UserPostListView(ListView):
    model = Post
    template_name = 'learningWebsite/user_posts.html'
    context_object_name = 'posts'
    paginate_by = 6
 
    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_created')
 
class PostDetailView(DetailView):
    context_object_name = 'learningWebsite/post_detail.html'
    template_name = 'learningWebsite/post_detail.html'
    queryset = Post.objects.all()

class PostCreateView(LoginRequiredMixin, CreateView):

    model = Post

    fields = ['title']
 
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
   
class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'good_training_images',
     'bad_training_images', 'good_validation_images', 
     'bad_validation_images', 'good_test_images', 
     'bad_test_images']
 
    def from_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
 
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
 
class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/'
 
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
 
def testing(request):
    return render(request, 'learningWebsite/testing.html',{'title': 'training'})
 
def output(request):
    return render(request, 'learningWebsite/output.html',{'title': 'output'})
 
 
def post_training(request):
 
    context = {
        'trainings': Training.objects.all()
    }
    return render(request, 'learningWebsite/training.html', context)

def TrainLoad(request):

    return render(request, 'learningWebsite/train_load.html')

def TrainingAlgorithm(request):

    train_dir = 'MachineLearning/Training/'
    vali_dir = 'MachineLearning/Validation/'
    test_dir = 'MachineLearning/Test/'

    train_malignant_dir = 'MachineLearning/Train/bad_training_images/'
    train_benign_dir = 'MachineLearning/Train/good_training_images/'
    vali_malignant_dir = 'MachineLearning/Validation/bad_validation_images/'
    vali_benign_dir = 'MachineLearning/Validation/good_validation_images/'
    test_malignant_dir = 'MachineLearning/Test/bad_test_images/'
    test_benign_dir = 'MachineLearning/Test/good_test_images/'

    goodPartNumber = 0
    badPartNumber = 0
    goodValNumber = 0
    badValNumber = 0
    goodTestNumber = 0
    badTestNumber = 0

    context = {
        'goodPartNumber': len(train_malignant_dir),
        'badPartNumber': len(train_benign_dir),
        'goodValNumber': len(vali_malignant_dir),
        'badValNumber': len(vali_benign_dir),
        'goodTestNumber': len(test_malignant_dir),
        'badTestNumber': len(test_benign_dir),
    }

    def build_model():
        model = models.Sequential()
                                    
        model.add(layers.Dense(512, activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=1e-4),
                    metrics=['acc'])
        return model

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            train_dir,              
            target_size=(150, 150),
            batch_size=20,               
            class_mode='binary')    
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
            vali_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    model = build_model()

    history = model.fit(
        train_generator,
        steps_per_epoch=92,     # 20 (batch_size) x 92 = 1840 train samples
        epochs=10,
        validation_data=validation_generator,
        validation_steps=26      # 20 (batch_size) x 26 = 520 validation samples
    )

    model.save('MachineLearning/TrainingAlgorithm/Model1.h5')

    return render(request, 'learningWebsite/Training.html', context)

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

class goodTrainingImageViewDelete(DeleteView):
    model = GoodTrainingImages
    success_url = '/'
 
    def test_func(self):
        image = self.get_object()
        True

def download_file(request):
    fl_path = 'MachineLearning/TrainingAlgorithm/Model1.h5'
    filename = 'DeepLearningModel.h5'
    fl = keras.models.load_model('MachineLearning/TrainingAlgorithm/Model1.h5')
    mime_type, _ = mimetypes.guess_type(fl_path)
    response = HttpResponse(fl, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

def EvaluateModel(request):
    test_dir = 'MachineLearning/Test/'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (150,150),
        batch_size = 267//20,
        class_mode = 'binary'
    )

    model = keras.models.load_model('MachineLearning/TrainingAlgorithm/Model1.h5')
    loss, acc = model.evaluate(test_generator, steps = 20)
    acc = acc * 100
    loss = loss * 100

    acc = round(acc, 2)
    loss = round(loss, 2)
    
    context = {
        'acc': acc,
        'loss': loss
    }

    return render(request, 'learningWebsite/Evaluation.html', context)

def LoadCamView(request):
    if request.method == "POST":
        form = LoadCamForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('load-model')
    
    else:
        form = LoadCamForm()
    return render(request, 'learningWebsite/LoadCam.html', {'form':form})


def LoadModelView(request):

    list_of_files = glob.glob('MachineLearning/UserTest/Image/*.bmp')
    image = max(list_of_files, key=os.path.getctime)
    photo = LoadModel.objects.create(image=image,)
    photo.save()

    def load(img):
        np_image = Image.open(img)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image,(150,150,3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    test_dir = 'MachineLearning/TrainingAlgorithm/Model1.h5'
    list_of_files = glob.glob('MachineLearning/UserTest/Image/*.bmp')
    img = max(list_of_files, key=os.path.getctime)
    img = load(img)
    model = keras.models.load_model(test_dir)
    guess = model.predict(img)

    guess = guess * 100
    guess = np.round(guess, 2)
    guess = guess[0][0]
    partType = ''

    if guess > 75:
        partType = 'Part has no defects'
    else:
        partType = 'Part has defects detected'

    context = {'guess': guess, 'label': partType, 'image': LoadModel.objects.all()}

    return render(request, 'learningWebsite/LoadModel.html', context)


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