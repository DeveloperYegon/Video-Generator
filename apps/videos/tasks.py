# apps/videos/tasks.py
import os
import uuid
import traceback
import logging
import redis
from contextlib import contextmanager
from django.core.files import File
from django.core.mail import send_mail
from django.conf import settings
from celery import shared_task
from .services import ScriptProcessor, MediaFinder, VoiceGenerator, VideoEditor
from .models import VideoProject, Scene, MediaAsset, AudioAsset, RenderedVideo,VideoScript


# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

logger = logging.getLogger(__name__)

@contextmanager
def redis_lock(lock_name, timeout=60):
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
    """
    Process a video project from script to rendered video.
    This task handles the entire pipeline:
    1. Process script into scenes
    2. Generate media for each scene
    3. Generate voiceovers 
    4. Render final video
    """

    logger.info(f"Starting processing for project {project_id}")
    
    try:
        # Get project and update status
        project = VideoProject.objects.get(id=project_id)
        project.status = 'processing'
        project.save()
        
        # Get script
        script = project.script
        if not script:
            logger.error(f"Project {project_id} has no script")
            raise Exception("Project has no script")
        
        # Process script into scenes
        logger.info(f"Processing script for project {project_id}")
        process_script_task(script.id)
        
        # Find media for scenes
        logger.info(f"Finding media for project {project_id}")
        find_media_task(script.id)
        
        # Generate voiceovers
        logger.info(f"Generating voiceovers for project {project_id}")
        generate_voiceovers_task(script.id)
        
        # Render video
        logger.info(f"Rendering video for project {project_id}")
        render_video_task(project_id)
        
        # Update project status
        project.refresh_from_db()
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
        
        logger.info(f"Completed processing for project {project_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing project {project_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        try:
            project = VideoProject.objects.get(id=project_id)
            project.status = 'failed'
            project.error_message = str(e)[:255]  # Truncate if needed
            project.save()
            
            # Notify user of failure
            if project.user.email:
                send_mail(
                    'Video processing failed',
                    f'Unfortunately, your video "{project.title}" could not be processed. Error: {str(e)}',
                    settings.DEFAULT_FROM_EMAIL,
                    [project.user.email],
                    fail_silently=True,
                )
        except Exception as inner_e:
            logger.error(f"Error updating project status: {str(inner_e)}")
        
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

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

@shared_task
def find_media_task(script_id):
    """Find media for all scenes in script"""

    try:
        script = VideoScript.objects.get(id=script_id)
        scenes = script.scenes.all().order_by('order')
        media_finder = MediaFinder()
        
        for scene in scenes:
            if scene.media_assets.exists():
                continue
                
            keywords = scene.keywords
            prompt = f"High quality YouTube background footage showing {' '.join(keywords)}, cinematic, 4k, trending on artstation"
            
            try:
                image_data = media_finder.generate_image(
                    prompt=prompt,
                    width=1024,
                    height=576
                )
                
                if image_data:
                    media_dir = os.path.join(settings.MEDIA_ROOT, 'media_assets')
                    os.makedirs(media_dir, exist_ok=True)
                    
                    filename = f"stability_{scene.id}_{'_'.join(keywords[:3])}.png"
                    filepath = os.path.join(media_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    MediaAsset.objects.update_or_create(
                        scene=scene,
                        defaults={
                            'asset_type': 'image',
                            'source': 'generated',
                            'file': f'media_assets/{filename}',
                            'duration_seconds': 5.0,
                            'generated_prompt': prompt
                        }
                    )
            except Exception as scene_e:
                logger.error(f"Error generating media for scene {scene.id}: {str(scene_e)}")
                continue
        
        return True
    except Exception as e:
        logger.error(f"Error finding media for script {script_id}: {str(e)}")
        logger.error(traceback.format_exc())
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
@shared_task(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=True)
def render_video_task(self, project_id):
    lock_name = f"render_video_task:project_{project_id}"
    try:
        with redis_lock(lock_name):
            logger.info(f"Starting video rendering for project {project_id}")
            project = VideoProject.objects.get(id=project_id)

            # Update project status
            project.status = 'processing'
            project.error_message = ''  # Clear previous errors
            project.save()

            script = project.script

            # Get all scenes for this project
            scenes_data = []
            scenes = Scene.objects.filter(script=script).order_by('order')
            scene_count = scenes.count()

            logger.info(f"Found {scene_count} scenes for project {project_id}")

            if scene_count == 0:
                raise Exception("No scenes found for the project")

            # Track missing assets for better error reporting
            scenes_missing_media = []
            scenes_missing_audio = []
            scenes_invalid_media = []
            scenes_invalid_audio = []

            for scene in scenes:
                media_asset = scene.media_assets.first()
                audio_asset = scene.audio_asset if hasattr(scene, 'audio_asset') else None

                # Check if media asset exists
                if not media_asset:
                    scenes_missing_media.append(scene.order)
                    continue

                # Check if audio asset exists
                if not audio_asset:
                    scenes_missing_audio.append(scene.order)
                    continue

                # Ensure file paths exist on disk
                if not os.path.exists(media_asset.file.path):
                    logger.error(f"Media file not found: {media_asset.file.name}")
                    scenes_missing_media.append(scene.order)
                    continue

                if not os.path.exists(audio_asset.file.path):
                    logger.error(f"Audio file not found: {audio_asset.file.name}")
                    scenes_missing_audio.append(scene.order)
                    continue

                # Verify file integrity using ffprobe
                media_info = VideoEditor.verify_media_file(media_asset.file.path)
                if not media_info:
                    logger.error(f"Invalid media file for scene {scene.order}: {media_asset.file.name}")
                    scenes_invalid_media.append(scene.order)
                    continue

                audio_info = VideoEditor.verify_media_file(audio_asset.file.path)
                if not audio_info:
                    logger.error(f"Invalid audio file for scene {scene.order}: {audio_asset.file.name}")
                    scenes_invalid_audio.append(scene.order)
                    continue

                # Add scene to rendering list
                scenes_data.append({
                    'media_path': media_asset.file.path,
                    'audio_path': audio_asset.file.path,
                    'text': scene.text,
                    'duration_seconds': media_asset.duration_seconds
                })

            # Report on missing assets
            if scenes_missing_media or scenes_missing_audio or scenes_invalid_media or scenes_invalid_audio:
                error_details = []

                if scenes_missing_media:
                    error_details.append(f"Missing media for scenes: {scenes_missing_media}")
                if scenes_missing_audio:
                    error_details.append(f"Missing audio for scenes: {scenes_missing_audio}")
                if scenes_invalid_media:
                    error_details.append(f"Invalid media files for scenes: {scenes_invalid_media}")
                if scenes_invalid_audio:
                    error_details.append(f"Invalid audio files for scenes: {scenes_invalid_audio}")

                if not scenes_data:
                    # Critical failure - can't render anything
                    error_message = f"Cannot render video: {'; '.join(error_details)}"
                    logger.error(error_message)

                    # Try to regenerate missing assets if applicable
                    if scenes_missing_media:
                        logger.info(f"Attempting to regenerate missing media assets")
                        find_media_task.delay(script.id)

                    if scenes_missing_audio:
                        logger.info(f"Attempting to regenerate missing audio assets")
                        generate_voiceovers_task.delay(script.id)

                    # Update project status
                    project.status = 'failed'
                    project.error_message = error_message[:255]  # Truncate if needed
                    project.save()

                    # Retry after delay (if we have retries left)
                    if self.request.retries < self.max_retries:
                        logger.info(
                            f"Scheduling retry ({self.request.retries + 1}/{self.max_retries}) for project {project_id}")
                        raise self.retry(countdown=60 * (2 ** self.request.retries))
                    else:
                        return False
                else:
                    # Non-critical - we can render with available scenes
                    logger.warning(
                        f"Rendering with partial scenes: {len(scenes_data)}/{scene_count} scenes available. {'; '.join(error_details)}")

            # Create output path
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos', f'project_{project_id}')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'video_{project_id}.mp4')

            # Log rendering attempt
            logger.info(f"Rendering video with {len(scenes_data)} scenes for project {project_id}")

            # Render the video with extended diagnostic info
            success, media_url = VideoEditor.combine_scenes(
                scenes_data=scenes_data,
                output_path=output_path,
                project_id=project_id
            )

            if not success:
                raise Exception("Video rendering failed. Check logs for details.")

            logger.info(f"Video rendered successfully for project {project_id}: {media_url}")

            # Get or create rendered video object
            RenderedVideo.objects.filter(project=project).delete()

            # Calculate total duration from scenes
            total_duration = sum(scene.get('duration_seconds', 5.0) for scene in scenes_data)

            # Create rendered video record
            RenderedVideo.objects.create(
                project=project,
                file=media_url,
                duration_seconds=total_duration,
                resolution='1920x1080'
            )

            # Update project status
            project.status = 'completed'
            project.error_message = ''  # Clear any error messages
            project.save()

            # Send notification email
            if project.user.email:
                send_mail(
                    'Your video is ready!',
                    f'Your video "{project.title}" has been successfully rendered and is now available for viewing.',
                    settings.DEFAULT_FROM_EMAIL,
                    [project.user.email],
                    fail_silently=True,
                )

            return True
    except Exception as e:
        logger.error(f"Error acquiring lock or rendering video for project {project_id}: {str(e)}")
        logger.error(traceback.format_exc())

        try:
            # Update project status
            project = VideoProject.objects.get(id=project_id)
            project.status = 'failed'
            project.error_message = str(e)[:255]  # Truncate if needed
            project.save()

            # Notify user of failure
            if project.user.email:
                send_mail(
                    'Video processing failed',
                    f'Unfortunately, your video "{project.title}" could not be processed. Error: {str(e)}',
                    settings.DEFAULT_FROM_EMAIL,
                    [project.user.email],
                    fail_silently=True,
                )
        except Exception as inner_e:
            logger.error(f"Error updating project status: {str(inner_e)}")

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            logger.info(f"Scheduling retry ({self.request.retries + 1}/{self.max_retries}) for project {project_id}")
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        else:
            return False