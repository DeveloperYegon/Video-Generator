from config.celery import shared_task
from .services import ScriptProcessor, MediaFinder, VoiceGenerator,VideoEditor
from django.core.mail import send_mail
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_video_project(self, project_id):
    from .models import VideoProject

    project = VideoProject.objects.get(id=project_id)
    project.status = 'processing'
    project.save()

    try:
        script = project.script

        # Process script into scenes
        processor = ScriptProcessor()
        sentences = processor.segment_script(script.original_text)

        # Clear existing scenes
        script.scenes.all().delete()

        # Create new scenes
        scenes = []
        for i, sentence in enumerate(sentences):
            keywords = processor.extract_keywords(sentence)
            scene = script.scenes.create(
                order=i + 1,
                text=sentence,
                keywords=keywords
            )
            scenes.append(scene)

        # Find media for each scene
        media_finder = MediaFinder()
        for scene in scenes:
            keywords = scene.keywords
            media = media_finder.find_media(keywords, media_type='video') or \
                    media_finder.find_media(keywords, media_type='photo')

            if media:
                # (Implement media download and saving as in views.py)
                pass

        # Generate voiceovers
        voice_gen = VoiceGenerator()
        for scene in scenes:
            # (Implement voiceover generation as in views.py)
            pass

        # Render video
        # (Implement video rendering as in views.py)

        project.status = 'completed'
        project.save()

        # Notify user
        if project.user.email:
            send_mail(
                'Your video is ready!',
                f'Your video "{project.title}" has been successfully processed.',
                settings.DEFAULT_FROM_EMAIL,
                [project.user.email],
                fail_silently=True,
            )

        return True

    except Exception as e:
        logger.error(f"Error processing project {project_id}: {str(e)}")
        project.status = 'failed'
        project.save()
        raise self.retry(exc=e, countdown=60, max_retries=3)