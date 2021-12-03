""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

from django import forms
from django.contrib.auth.models import User
from .models import *
from django.forms import ClearableFileInput
 
class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title',
        'good_training_images',
            'bad_training_images', 'good_validation_images', 
            'bad_validation_images', 'good_test_images', 
            'bad_test_images'
            ]
        widgets = {
            'good_training_images': ClearableFileInput(attrs={'multiple':True}),
            'bad_training_images': ClearableFileInput(attrs={'multiple':True}),
            'good_validation_images': ClearableFileInput(attrs={'multiple':True}),
            'bad_validation_images': ClearableFileInput(attrs={'multiple':True}),
            'good_test_images': ClearableFileInput(attrs={'multiple':True}),
            'bad_test_images': ClearableFileInput(attrs={'multiple':True}),
        }

class PostFullForm(PostForm):
    images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple':True}))

    class Meta(PostForm.Meta):
        fields = PostForm.Meta.fields + ['images', ]

class LoadCamForm(forms.ModelForm):

    class Meta:
        model = LoadCam
        fields = ['dlmodel', 'camera']

class LoadModelForm(forms.ModelForm):
    file = forms.FileField()
    class Meta:
        model = LoadModel
        fields = ['image']

