# Generated by Django 3.2.8 on 2021-10-29 14:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learningWebsite', '0009_auto_20211029_0347'),
    ]

    operations = [
        migrations.CreateModel(
            name='EvaluateImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='MachineLearning/EvaluateImage/')),
            ],
        ),
    ]