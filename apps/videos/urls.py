from django.urls import path
from . import views

app_name = 'videos'

urlpatterns = [
    path('create/', views.create_project, name='create_project'),
    path('<int:project_id>/', views.view_project, name='view_project'),
    path('<int:project_id>/edit-script/', views.edit_script, name='edit_script'),
    path('<int:project_id>/process-script/', views.process_script, name='process_script'),
    path('<int:project_id>/find-media/', views.find_media, name='find_media'),
    path('voiceovers/<int:project_id>/generate-voiceovers/', views.generate_voiceovers, name='generate_voiceovers'),
    path('<int:project_id>/render-video/', views.render_video, name='render_video'),
    path('<int:project_id>/generate-script/', views.generate_script, name='generate_script'),
    path('', views.project_list, name='project_list'),
]