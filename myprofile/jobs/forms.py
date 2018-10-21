from django import forms
from jobs.models import Job

class Editjob(forms.ModelForm):
	class Meta:
		model = Job
		fields = '__all__'