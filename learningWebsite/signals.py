""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

# this file is where signals from the 'Users' folder are kept. 
# if a model within the learningWebsite is in need of a User.id or is saving a profile,
# this is where they would be directed to

from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import *
 
 
@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Post.objects.create(user=instance)
 
 
@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()
 
 
# @receiver(post_save, sender=User)
# def create_good_training(sender, instance, created, **kwargs):
#     instance.GoodTrainingImage.save()
 
# @receiver(post_save, sender=User)
# def save_good_training(sender, instance, **kwargs):
#     instance.GoodTrainingImage.save()

# @receiver(post_save, sender=User)
# def create_bad_training(sender, instance, created, **kwargs):
#     instance.BadTrainingImage.save()
 
 
# @receiver(post_save, sender=User)
# def save_bad_training(sender, instance, **kwargs):
#     instance.BadTrainingImage.save()

# @receiver(post_save, sender=User)
# def create_CamImg(sender, instance, created, **kwargs):
#     instance.LoadModel.save()

# @receiver(post_save, sender=User)
# def save_CamImg(sender, instance, **kwargs):
#     instance.LoadModel.save()