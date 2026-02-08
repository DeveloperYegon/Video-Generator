import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')

app = Celery('videomaker')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Use broker/backend from Django settings (so Docker/localhost fix in base.py is applied)
from django.conf import settings
redis_url = getattr(settings, 'CELERY_BROKER_URL', None) or os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
app.conf.update(
    broker_url=redis_url,
    result_backend=getattr(settings, 'CELERY_RESULT_BACKEND', None) or redis_url,
    broker_connection_retry_on_startup=True
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
