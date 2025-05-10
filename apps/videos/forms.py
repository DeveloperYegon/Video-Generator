from django import forms
from .models import VideoProject, VideoScript

class VideoProjectForm(forms.ModelForm):
    class Meta:
        model = VideoProject
        fields = ['title', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class ScriptForm(forms.ModelForm):
    class Meta:
        model = VideoScript
        fields = ['original_text']
        widgets = {
            'original_text': forms.Textarea(attrs={'rows': 15, 'class': 'form-control'}),
        }