""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

from django.db import models
from django.contrib.auth.models import User
from PIL import Image
 
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')
 
    def __str__(self):
        return f'{self.user.username} Profile'
 
    # def save(self, *args, **kwargs):
    #     super(Profile, self).save(*args, **kwargs)
 
    #     img = Image.open(self.image.path)
    #     rgb_im = img.convert('RGB')
    #     rgb_im.save('audacious.jpg')
 
    #     if rgb_im.height > 300 or rgb_im.width > 300:
    #         output_size = (300, 300)
    #         rgb_im.thumbnail(output_size)
    #         rgb_im.save(self.image.path)
 
   
