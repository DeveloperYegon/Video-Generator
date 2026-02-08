#\Video-Generator-main\config\settings\dev.py
from .base import *
import socket
# Debug settings
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '[::1]']
#SILENCED_SYSTEM_CHECKS = ['debug_toolbar.W004']

# Django Debug Toolbar
#INSTALLED_APPS += ['debug_toolbar']
#MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')

# Debug Toolbar configuration
#DEBUG_TOOLBAR_CONFIG = {
 #   'SHOW_TOOLBAR_CALLBACK': lambda request: True,
#}

# Configure internal IPs for Debug Toolbar
#hostname, _, ips = socket.gethostbyname_ex(socket.gethostname())
#INTERNAL_IPS = [ip[: ip.rfind(".")] + ".1" for ip in ips] + ["127.0.0.1",'localhost', "10.0.2.2"]

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT'),
    }
}


# Email
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Static files
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'