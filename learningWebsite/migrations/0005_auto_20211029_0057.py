# Generated by Django 3.2.8 on 2021-10-29 04:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learningWebsite', '0004_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='bad_training_images',
            field=models.ImageField(blank=True, upload_to='bad_training_images'),
        ),
        migrations.AlterField(
            model_name='post',
            name='good_training_images',
            field=models.ImageField(blank=True, upload_to='good_training_images'),
        ),
    ]