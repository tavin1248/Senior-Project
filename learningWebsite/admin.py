""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """


from django.contrib import admin
from .models import Post, GoodTrainingImages, BadTrainingImages, GoodValidationImages, BadValidationImages, GoodTestImages,BadTestImages, LoadModel
 
admin.site.register(Post)
admin.site.register(GoodTrainingImages)
admin.site.register(BadTrainingImages)
admin.site.register(GoodValidationImages)
admin.site.register(BadValidationImages)
admin.site.register(GoodTestImages)
admin.site.register(BadTestImages)
admin.site.register(LoadModel)