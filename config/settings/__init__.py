import os
from .base import *

ENVIRONMENT = os.getenv('DJANGO_ENV', 'dev')

if ENVIRONMENT == 'prod':
    from .prod import *
elif ENVIRONMENT == 'ci':
    from .ci import *
else:
    from .dev import *