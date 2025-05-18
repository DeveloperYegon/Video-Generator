# apps/videos/forms.py
from django import forms
from .models import VideoProject, VideoScript
from .services import VoiceGenerator


class VideoProjectForm(forms.ModelForm):
    class Meta:
        model = VideoProject
        fields = ['title', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }
        
class ScriptForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 15, 'class': 'form-control'}),
        required=True
    )
    voice_id = forms.ChoiceField(
        label='Voice',
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=True
    )
    
    def __init__(self, *args, **kwargs):
        voice_choices = kwargs.pop('voice_choices', None)
        super(ScriptForm, self).__init__(*args, **kwargs)
        
        if not voice_choices:
            # Get available voices if not provided
            voice_gen = VoiceGenerator()
            voices = voice_gen.get_available_voices()
            voice_choices = [(voice_id, f"{voice_id} - {description}") for voice_id, description in voices.items()]
        
        self.fields['voice_id'].choices = voice_choices