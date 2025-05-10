from django.apps import AppConfig
import os


class VideosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.videos'
    def ready(self):
        # Run NLTK check on startup in development
        if os.getenv('DJANGO_ENV') == 'dev':
            from django.core.management import call_command
            call_command('nltk_check')
    