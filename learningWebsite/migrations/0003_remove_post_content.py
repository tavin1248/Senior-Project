# Generated by Django 3.2.8 on 2021-10-29 02:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('learningWebsite', '0002_auto_20211028_2236'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='content',
        ),
    ]