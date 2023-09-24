# phishing_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Associate 'index' view with the root URL
    path('result/', views.result, name='result'),
]
