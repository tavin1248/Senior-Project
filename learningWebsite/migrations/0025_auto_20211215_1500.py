# Generated by Django 3.2.8 on 2021-12-15 20:00

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('learningWebsite', '0024_alter_goodtrainingimages_reference_post'),
    ]

    operations = [
        migrations.AddField(
            model_name='goodtrainingimages',
            name='author',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='auth.user'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='goodtrainingimages',
            name='reference_post',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='learningWebsite.post', verbose_name='Post'),
        ),
    ]