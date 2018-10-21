from django.shortcuts import render,redirect
from jobs.models import Job
from jobs.forms import Editjob
from django.contrib.auth.models import User
from django.http import HttpResponse





def home(request):
	jobs=Job.objects.all()
	return render(request, 'jobs/home.html',{'jobs':jobs})



def edit(request,id=None):
	try:
		current_job=Job.objects.get(id=id)

		if request.method == 'POST':
			form=Editjob(request.POST,request.FILES,instance=current_job)
			if form.is_valid():
				form.save()
				return redirect('home')
		else:
			form=Editjob(instance=current_job)
			return render (request , 'jobs/edit.html',{'form':form})
	except:
		return HttpResponse ('Please try again something went wrong.')



def add(request):
	try:
		if request.method == 'POST':
			form = Editjob(request.POST,request.FILES)
			if form.is_valid():
				form.save()
				return redirect('home')
			else:
				return HttpResponse ('Not valid')
		else:
			form = Editjob()
			return render (request ,'jobs/edit.html', {'form':form})
	except:
		return HttpResponse ('Please try again something went wrong.')













# Create your views here.
