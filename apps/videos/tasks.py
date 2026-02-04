# apps/videos/tasks.py
import os
import uuid
import traceback
import logging
import redis
import shutil  # MODIFIED: Added for moving FLUX files
from contextlib import contextmanager
from django.core.files import File
from django.core.mail import send_mail
from django.conf import settings
from celery import shared_task, chain  # MODIFIED: Added chain
from .services import ScriptProcessor, MediaFinder, VoiceGenerator, VideoEditor
from .models import VideoProject, Scene, MediaAsset, AudioAsset, RenderedVideo,VideoScript


# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

logger = logging.getLogger(__name__)

@contextmanager
def redis_lock(lock_name, timeout=1800):  # MODIFIED: Extended to 30 mins
    """Acquire a Redis lock with a timeout."""
    lock = redis_client.lock(lock_name, timeout=timeout)
    try:
        acquired = lock.acquire(blocking_timeout=10)
        if not acquired:
            raise Exception(f"Could not acquire lock: {lock_name}")
        yield
    finally:
        if lock.locked():
            lock.release()


@shared_task(bind=True, max_retries=3)
def process_video_project(self, project_id):
    """Orchestrates the pipeline using Celery Chains
    Process a video project from script to rendered video.
    This task handles the entire pipeline:
    1. Process script into scenes
    2. Generate media for each scene
    3. Generate voiceovers 
    4. Render final video
    """

    logger.info(f"Starting processing for project {project_id}")
    
    try:
        project = VideoProject.objects.get(id=project_id)
        project.status = 'processing'
        project.save()

        # MODIFIED: Sequential pipeline with high timeouts
        pipeline = chain(
            process_script_task.s(project.script.id),
            find_media_task.s(project.script.id),
            generate_voiceovers_task.s(project.script.id),
            render_video_task.s(project_id)
        )
        pipeline.apply_async()
        return True
    except Exception as e:
        project.status = 'failed'
        project.save()
        raise self.retry(exc=e, countdown=60)

@shared_task
def process_script_task(script_id):
    """Process script into scenes with keywords"""

    try:
        script = VideoScript.objects.get(id=script_id)
        processor = ScriptProcessor()
        
        # Segment script into scenes
        sentences = processor.segment_script(script.original_text)
        
        # Clear existing scenes
        Scene.objects.filter(script=script).delete()
        
        # Create new scenes
        for i, sentence in enumerate(sentences):
            keywords = processor.extract_keywords(sentence)
            Scene.objects.create(
                script=script,
                order=i + 1,
                text=sentence,
                keywords=keywords
            )
        
        return True
    except Exception as e:
        logger.error(f"Error processing script {script_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@shared_task(time_limit=600)  # MODIFIED: 10 min limit for FLUX
def find_media_task(script_id):
    """Find media for all scenes in script"""

    try:
        script = VideoScript.objects.get(id=script_id)
        scenes = script.scenes.all().order_by('order')
        media_finder = MediaFinder()

        for scene in scenes:
            # MODIFIED: Stop if project was marked failed/cancelled
            if VideoProject.objects.filter(script=script.project, status='failed').exists():
                return False

            prompt = f"Cinematic 4k, {' '.join(scene.keywords)}"
            # MODIFIED: MediaFinder now returns a local path
            temp_path = media_finder.generate_image(prompt)

            if temp_path and os.path.exists(temp_path):
                media_dir = os.path.join(settings.MEDIA_ROOT, 'media_assets')
                os.makedirs(media_dir, exist_ok=True)

                # MODIFIED: Move file from Gradio temp to Django Media
                ext = os.path.splitext(temp_path)[1]
                filename = f"flux_{scene.id}{ext}"
                final_path = os.path.join(media_dir, filename)
                shutil.move(temp_path, final_path)

                MediaAsset.objects.create(
                    scene=scene,
                    file=f'media_assets/{filename}',
                    asset_type='image'
                )
        return True
    except Exception as e:
        logger.error(f"Media Task Error: {str(e)}")
        raise

@shared_task
def generate_voiceovers_task(script_id):
    """Generate voiceovers for all scenes in script"""

    try:
        script = VideoScript.objects.get(id=script_id)
        scenes = script.scenes.all().order_by('order')
        voice_gen = VoiceGenerator()
        
        # Get the selected voice ID from the script
        voice_id = script.voice_id
        
        # Remove existing audio assets
        AudioAsset.objects.filter(scene__script=script).delete()
        
        for scene in scenes:
            # Generate voiceover with the selected voice
            audio_file = voice_gen.generate_voiceover(scene.text, voice_id=voice_id)
            
            # Skip if no audio was generated (e.g., empty text after cleaning)
            if not audio_file:
                continue
                
            # Create audio asset
            with open(audio_file, 'rb') as f:
                audio_asset = AudioAsset.objects.create(
                    scene=scene,
                    voice_id=voice_id
                )
                audio_asset.file.save(f'{uuid.uuid4()}.mp3', File(f))
                
            # Clean up
            os.unlink(audio_file)
        
        return True
    except Exception as e:
        logger.error(f"Error generating voiceovers for script {script_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise
#@shared_task(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=True)
@shared_task(bind=True, max_retries=3)
def render_video_task(self, project_id):
    from .models import VideoProject, Scene, MediaAsset
    from .services import VideoEditor
    import logging
    import os
    from django.core.mail import send_mail
    from django.conf import settings

    logger = logging.getLogger(__name__)
    logger.info(f"Starting video rendering for project {project_id}")

    try:
        project = VideoProject.objects.get(id=project_id)
    except VideoProject.DoesNotExist:
        logger.error(f"VideoProject with id {project_id} does not exist")
        return False

    scenes = Scene.objects.filter(script__project=project).order_by('order')
    logger.info(f"Found {len(scenes)} scenes for project {project_id}")

    scenes_data = []
    scenes_missing_media = []

    for scene in scenes:
        media_asset = scene.media_assets.first()
        audio_asset = scene.audio_asset if hasattr(scene, 'audio_asset') else None

        if not media_asset or not os.path.exists(media_asset.file.path):
            scenes_missing_media.append(scene.order)
            continue

        if not audio_asset or not os.path.exists(audio_asset.file.path):
            scenes_missing_media.append(scene.order)
            continue

        media_info = VideoEditor.verify_media_file(media_asset.file.path)
        if not media_info:
            logger.warning(f"Invalid media file for scene {scene.order}: {media_asset.file.name}")
            scenes_missing_media.append(scene.order)
            continue

        audio_info = VideoEditor.verify_media_file(audio_asset.file.path)
        if not audio_info:
            logger.warning(f"Invalid audio file for scene {scene.order}: {audio_asset.file.name}")
            scenes_missing_media.append(scene.order)
            continue

        scenes_data.append({
            'media_path': media_asset.file.path,
            'audio_path': audio_asset.file.path,
            'text': scene.text,
            'duration_seconds': media_asset.duration_seconds
        })

    if scenes_missing_media:
        logger.error(f"Cannot render video: Missing media for scenes: {scenes_missing_media}")
        logger.info("Attempting to regenerate missing media assets")
        from .tasks import find_media_task, generate_voiceovers_task
        script = project.script
        find_media_task.delay(script.id)
        generate_voiceovers_task.delay(script.id)
        project.status = 'failed'
        project.error_message = f"Missing media for scenes: {scenes_missing_media}"
        project.save()
        if project.user.email:
            send_mail(
                'Video processing failed',
                f'Unfortunately, your video "{project.title}" could not be processed. Error: {project.error_message}',
                settings.DEFAULT_FROM_EMAIL,
                [project.user.email],
                fail_silently=True,
            )
        raise self.retry(countdown=60 * (2 ** self.request.retries))

    if not scenes_data:
        error_message = "No valid scenes available for rendering"
        logger.error(error_message)
        project.status = 'failed'
        project.error_message = error_message
        project.save()
        if project.user.email:
            send_mail(
                'Video processing failed',
                f'Unfortunately, your video "{project.title}" could not be processed. Error: {error_message}',
                settings.DEFAULT_FROM_EMAIL,
                [project.user.email],
                fail_silently=True,
            )
        return False

    logger.info(f"Rendering video with {len(scenes_data)} scenes for project {project_id}")
    output_path = os.path.join(settings.MEDIA_ROOT, 'rendered_videos', f'project_{project_id}', f'video_{project_id}.mp4')

    try:
        # Call the correct method: combine_videos instead of combine_scenes
        success = VideoEditor.combine_videos(project_id, scenes_data, output_path)
        if success:
            media_url = os.path.join(settings.MEDIA_URL, 'rendered_videos', f'project_{project_id}', f'video_{project_id}.mp4')
            project.status = 'completed'
            project.media_url = media_url
            project.error_message = ''
            project.save()
            logger.info(f"Updated project {project_id} status to 'completed'")
            return True
        else:
            raise Exception("Video rendering failed. Check logs for details.")
    except Exception as e:
        logger.error(f"Error acquiring lock or rendering video for project {project_id}: {str(e)}")
        project.status = 'failed'
        project.error_message = str(e)[:255]
        project.save()
        logger.info(f"Updated project {project_id} status to 'failed'")
        if project.user.email:
            send_mail(
                'Video processing failed',
                f'Unfortunately, your video "{project.title}" could not be processed. Error: {str(e)}',
                settings.DEFAULT_FROM_EMAIL,
                [project.user.email],
                fail_silently=True,
            )
        if self.request.retries < self.max_retries:
            logger.info(f"Scheduling retry ({self.request.retries + 1}/{self.max_retries}) for project {project_id}")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        return False