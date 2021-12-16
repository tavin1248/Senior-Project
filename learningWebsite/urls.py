""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

#This file contains all of the url paths for 'learningWebsite'
# If an html or redirect page calls any of the 'name=****' functions they will be redirected
# to a specific view and the url will be updated to the tag on the left. 

from django.urls import path
from . import views
from .views import *
 
urlpatterns = [
    path('', PostListView.as_view(), name='learningWebsite-home'),
    path('user/<str:username>/', UserPostListView.as_view(), name='user-posts'),
    path('post/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('post/new/', PostCreateView.as_view(), name='post-create'),
    path('post/<int:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('post/<int:pk>/delete/', PostDeleteView.as_view(), name='post-delete'),
    path('goodTraining/delete/', goodTrainingImageViewDelete.as_view(), name='good-training-image-delete'),
    

    path('goodTraining/', views.goodTrainingImageView, name='good-training-image'),
    path('badTraining/', views.badTrainingImageView, name='bad-training-image'),
    path('goodValidation/', views.goodValidationImageView, name='good-validation-image'),
    path('badValidation/', views.badValidationImageView, name='bad-validation-image'),
    path('goodTest/', views.goodTestImageView, name='good-test-image'),
    path('badTest/', views.badTestImageView, name='bad-test-image'),
    path('Evaluate/', views.EvaluateModel, name='evaluate'),
    path('Train/', views.TrainingAlgorithm, name='Train'),
    path('TrainLoad/', views.TrainLoad, name='train-load'),
    path('LoadModel/', views.LoadModelView, name='load-model'),
    path('LoadCam/', views.LoadCamView, name='load-cam'),
    path('<str:MachineLearning/TrainingAlgorithm/Model1.h5/', views.download_file, name='download-model'), #This specific url is for downloading the .h5 file
    path('HowTo/', views.HowToView, name='how-to'),
    path('testing/', views.testing,name='learningWebsite-testing'),
    path('output/', views.output,name='learningWebsite-output')
]