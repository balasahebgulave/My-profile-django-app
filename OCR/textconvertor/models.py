from django.db import models


class ImageViewer(models.Model):
	image=models.ImageField(upload_to='',blank=True)

# Create your models here.
