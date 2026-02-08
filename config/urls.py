
from django.contrib import admin
from django.urls import path, include
from django.conf import settings

# Import static to serve media files in development
from django.conf.urls.static import static

urlpatterns = [
    path('', include(('apps.videos.urls', 'videos'), namespace='videos')),
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
]



# Only add debug toolbar if in DEBUG mode AND the package exists
if settings.DEBUG:
    try:
        import debug_toolbar
        urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]
    except ImportError:
        pass

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
#import debug_toolbar
#if settings.DEBUG:
  #  urlpatterns = [
    #    path('__debug__/', include(debug_toolbar.urls)),
    # + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + urlpatterns