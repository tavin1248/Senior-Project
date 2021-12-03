""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse
from django.views.generic.dates import timezone_today
from django.core.validators import FileExtensionValidator
 
class Post(models.Model):
 
    title = models.CharField(max_length=100)

    good_training_images = models.ImageField(blank=True, upload_to='good_training_images')
    bad_training_images = models.ImageField(blank=True,upload_to='bad_training_images')

    good_validation_images = models.ImageField(blank=True,upload_to='good_validation_pics')
    bad_validation_images = models.ImageField(blank=True,upload_to='bad_validation_pics')

    good_test_images = models.ImageField(blank=True,upload_to='good_test_pics')
    bad_test_images = models.ImageField(blank=True,upload_to='bad_test_pics')

    date_created = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
 
    def __str__(self):
        return self.title
 
    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk': self.pk})
 
class GoodTrainingImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Training/good_training_images/')

    def __str___(self):
        return self.image

class BadTrainingImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Training/bad_training_images/')

    def __str___(self):
        return self.image

class GoodValidationImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Validation/good_validation_images/')

    def __str___(self):
        return self.image

class BadValidationImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Validation/bad_validation_images/')

    def __str___(self):
        return self.image

class GoodTestImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Test/good_test_images/')

    def __str___(self):
        return self.image

class BadTestImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Test/bad_test_images/')

    def __str___(self):
        return self.image

class LoadCam(models.Model):
    dlmodel = models.FileField(default=None, upload_to='MachineLearning/UserTest/Model/', validators=[FileExtensionValidator(allowed_extensions=['h5'])])
    camera = models.CharField(max_length=100)

class LoadModel(models.Model):
    image = models.ImageField(upload_to='MachineLearning/UserTest/Image/')

class Images(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='good_training_pics',null=True,blank=True)
