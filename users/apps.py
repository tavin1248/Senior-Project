""" Tavin Ardell
    000754847
    "I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own" """

from django.apps import AppConfig
 
 
class UsersConfig(AppConfig):
    name = 'users'
 
    def ready(self):
        import users.signals
