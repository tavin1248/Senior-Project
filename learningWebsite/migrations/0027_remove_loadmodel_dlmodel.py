# Generated by Django 3.2.8 on 2021-12-16 08:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('learningWebsite', '0026_loadmodel_dlmodel'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='loadmodel',
            name='dlmodel',
        ),
    ]
