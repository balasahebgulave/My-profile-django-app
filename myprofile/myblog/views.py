from django.shortcuts import render
from myblog.models import Blog



def bloghome(request):
	blogs = Blog.objects.all()
	return render(request, 'blogs/bloghome.html',{'blogs':blogs})


def detail(request, id=None):
	blog=Blog.objects.get(id=id)
	return render(request,'blogs/detail.html',{'blog':blog})

# Create your views here.
