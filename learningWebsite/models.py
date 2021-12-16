""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

# This file is where all the models within learningWebsite are kept. 'Models' in this instance are django's class types.
# All ImageField files are automatically uploaded to the amazon s3 bucket associated with this project.

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse
from django.views.generic.dates import timezone_today
from django.core.validators import FileExtensionValidator
from django.db import models

# This model is the basic structure for a singular project post.
# It allows the user to enter a title and it will auto generate 
# the date the project was created and what user created the project.
class Post(models.Model):
 
    title = models.CharField(max_length=100)
    date_created = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name_plural = "Post"

    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk': self.pk})

# This model acts as a location to store the good training images needed to train the deep learning model
class GoodTrainingImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Training/good_training_images/')
    #reference_post = models.ForeignKey(Post, default=1 ,verbose_name="Post", on_delete=models.CASCADE)
    #author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str___(self):
        return self.image
# This model acts as a location to store the bad training images needed to train the deep learning model
class BadTrainingImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Training/bad_training_images/')
    #post = models.ForeignKey(Post, default=1, verbose_name="Post", on_delete=models.CASCADE)


    def __str___(self):
        return self.image
# This model acts as a location to store the good validation images needed to validate the deep learning model
class GoodValidationImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Validation/good_validation_images/')
    #post = models.ForeignKey(Post, default=1, verbose_name="Post", on_delete=models.CASCADE)

    def __str___(self):
        return self.image

# This model acts as a location to store the bad validation images needed to validate the deep learning model
class BadValidationImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Validation/bad_validation_images/')
    #post = models.ForeignKey(Post, default=1, verbose_name="Post", on_delete=models.CASCADE)

    def __str___(self):
        return self.image

# This model acts as a location to store the good test images needed to test the deep learning model
class GoodTestImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Test/good_test_images/')
    #post = models.ForeignKey(Post, default=1, verbose_name="Post", on_delete=models.CASCADE)

    def __str___(self):
        return self.image

# This model acts as a location to store the good test images needed to test the deep learning model
class BadTestImages(models.Model):
    image = models.ImageField(upload_to='MachineLearning/Test/bad_test_images/')
    #post = models.ForeignKey(Post, default=1, verbose_name="Post", on_delete=models.CASCADE)

    def __str___(self):
        return self.image

# This model is where a user entered deep learning model and camera IP address are stored
class LoadCam(models.Model):
    dlmodel = models.FileField(default=None, upload_to='MachineLearning/UserTest/Model/', validators=[FileExtensionValidator(allowed_extensions=['h5'])])
    camera = models.CharField(max_length=100)

# This model is where the camera input image is stored
class LoadModel(models.Model):
    image = models.ImageField(upload_to='MachineLearning/UserTest/Image/')

# This model is necessary for multiple image uploads
class Images(models.Model):
    image = models.ImageField(upload_to='good_training_pics',null=True,blank=True)