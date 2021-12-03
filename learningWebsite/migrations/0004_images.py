# Generated by Django 3.2.8 on 2021-10-29 03:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('learningWebsite', '0003_remove_post_content'),
    ]

    operations = [
        migrations.CreateModel(
            name='Images',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(blank=True, null=True, upload_to='good_training_pics')),
                ('post', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='learningWebsite.post')),
            ],
        ),
    ]
