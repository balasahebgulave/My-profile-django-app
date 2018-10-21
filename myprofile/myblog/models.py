from django.db import models

class Blog(models.Model):
	title=models.CharField(max_length=200)
	published=models.DateTimeField()
	image=models.ImageField(upload_to='media')
	body=models.TextField()

	def __str__(self):
		return self.title

	def summary(self):
		return self.body[0:500]


# Create your models here.
