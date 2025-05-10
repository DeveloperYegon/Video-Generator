from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


class VideoProject(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, default='draft',
                              choices=[
                                  ('draft', 'Draft'),
                                  ('processing', 'Processing'),
                                  ('completed', 'Completed'),
                                  ('failed', 'Failed')
                              ])

    def __str__(self):
        return self.title


class VideoScript(models.Model):
    project = models.OneToOneField(VideoProject, on_delete=models.CASCADE, related_name='script')
    original_text = models.TextField()
    generated_by_ai = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Script for {self.project.title}"


class Scene(models.Model):
    script = models.ForeignKey(VideoScript, on_delete=models.CASCADE, related_name='scenes')
    order = models.PositiveIntegerField()
    text = models.TextField()
    keywords = models.JSONField(default=list)  # Stores extracted keywords

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"Scene {self.order} - {self.text[:50]}..."


class MediaAsset(models.Model):
    scene = models.ForeignKey(Scene, on_delete=models.CASCADE, related_name='media_assets')
    asset_type = models.CharField(max_length=10, choices=[('image', 'Image'), ('video', 'Video')])
    source = models.CharField(max_length=20, default='pexels',
                              choices=[('pexels', 'Pexels'), ('upload', 'Uploaded')])
    url = models.URLField(blank=True)
    file = models.FileField(upload_to='media_assets/', blank=True, null=True,
                            validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'mp4'])])
    duration_seconds = models.FloatField(default=5.0)

    def __str__(self):
        return f"{self.get_asset_type_display()} for {self.scene}"


class AudioAsset(models.Model):
    scene = models.OneToOneField(Scene, on_delete=models.CASCADE, related_name='audio_asset')
    file = models.FileField(upload_to='audio_assets/')
    voice_id = models.CharField(max_length=50)  # Polly voice ID
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio for {self.scene}"


class RenderedVideo(models.Model):
    project = models.OneToOneField(VideoProject, on_delete=models.CASCADE, related_name='rendered_video')
    file = models.FileField(upload_to='rendered_videos/')
    created_at = models.DateTimeField(auto_now_add=True)
    duration_seconds = models.FloatField()
    resolution = models.CharField(max_length=20)

    def __str__(self):
        return f"Rendered video for {self.project.title}"