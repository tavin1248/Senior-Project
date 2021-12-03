""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Profile
 
 
@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
 
 
@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()
