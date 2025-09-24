from django.urls import path
from . import views

urlpatterns = [
    path('predict-emotion', views.predict_emotion, name='predict_emotion'),
]