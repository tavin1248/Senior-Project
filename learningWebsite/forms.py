""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

# this 'forms' file allows be to used django's 'form' function which are extremely 
# useful a user is inputting information within an html pages

from django import forms
from django.contrib.auth.models import User
from .models import *
from django.forms import ClearableFileInput

#form for the project name
class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title']

#this form is needed in order to upload multiple images to various models
class PostFullForm(PostForm):
    images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple':True}))

    class Meta(PostForm.Meta):
        fields = PostForm.Meta.fields + ['images', 'title']

#the LoadCamForm allows a user to input a deeplearning model and an IP address into an html page
class LoadCamForm(forms.ModelForm):

    class Meta:
        model = LoadCam
        fields = ['dlmodel', 'camera']

#The loadModelForm allows the backend program to upload the image from the camera
class LoadModelForm(forms.ModelForm):

    # file = forms.FileField()

    class Meta:
        model = LoadModel
        fields = ['image']

