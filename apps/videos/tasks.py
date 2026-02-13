import os
import uuid
import traceback
import logging
import shutil
import redis
from contextlib import contextmanager
import time

import httpx
from django.core.files import File
from django.core.mail import send_mail
from django.conf import settings
from django.db import transaction # ADDED: For database integrity
from celery import shared_task, chain

from .services import ScriptProcessor, MediaFinder, VoiceGenerator, VideoEditor
from .models import VideoProject, Scene, MediaAsset, AudioAsset, RenderedVideo, VideoScript

# MODIFIED: Define the variable FIRST, then use it.
# This ensures it works on your laptop (via 'redis' hostname) and avoids NameError.
REDIS_HOST = os.getenv('REDIS_HOST', 'redis') 

# Initialize Redis client correctly
redis_client = redis.Redis(
    host=REDIS_HOST, 
    port=6379, 
    db=0
)

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def process_video_project(self, project_id):
    """Orchestrates the pipeline using Celery Chains"""
    logger.info(f"Starting processing for project {project_id}")
    
    try:
        project = VideoProject.objects.get(id=project_id)
        project.status = 'processing'
        project.save()

        # Sequential chain: ensures data exists before next step starts
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
        logger.error(f"Pipeline initiation failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)

@shared_task
def process_script_task(script_id):
    """Process script into scenes with keywords"""
    try:
        # Atomic transaction ensures scenes are fully created before chain proceeds
        with transaction.atomic():
            script = VideoScript.objects.get(id=script_id)
            processor = ScriptProcessor()
            sentences = processor.segment_script(script.original_text)
            
            Scene.objects.filter(script=script).delete()
            
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
        raise

@shared_task(time_limit=600)
def find_media_task(script_id):
    """Find media for all scenes in script via FLUX API"""
    try:
        script = VideoScript.objects.get(id=script_id)
        scenes = script.scenes.all().order_by('order')
        media_finder = MediaFinder()
        saved_count = 0

        for scene in scenes:
            prompt = f"Cinematic 4k, {' '.join(scene.keywords)}"
            try:
                temp_path = media_finder.generate_image(prompt)
                time.sleep(10)
            except (httpx.ConnectError, OSError) as e:
                errno = getattr(e, 'errno', None)
                if errno == -5 or 'hostname' in str(e).lower() or 'address' in str(e).lower():
                    raise RuntimeError(
                        "Cannot reach Hugging Face FLUX API (DNS/network). "
                        "From Docker: ensure the worker has outbound internet and DNS (e.g. in docker-compose add: dns: [8.8.8.8] under worker). "
                        "Check HF_API_KEY and firewall/proxy."
                    ) from e
                raise

            if temp_path and os.path.exists(temp_path):
                media_dir = os.path.join(settings.MEDIA_ROOT, 'media_assets')
                os.makedirs(media_dir, exist_ok=True)

                ext = os.path.splitext(temp_path)[1]
                filename = f"flux_{scene.id}_{uuid.uuid4().hex[:6]}{ext}"
                final_path = os.path.join(media_dir, filename)
                
                # CLOUD FIX: In Docker, moving between /tmp and /app/media 
                # can be a cross-device link error. Fallback to copy.
                try:
                    shutil.move(temp_path, final_path)
                except OSError:
                    shutil.copy(temp_path, final_path)
                    os.remove(temp_path)

                MediaAsset.objects.update_or_create(
                    scene=scene,
                    defaults={
                        'file': f'media_assets/{filename}',
                        'asset_type': 'image'
                    }
                )
                saved_count += 1

        if saved_count == 0:
            logger.error("find_media_task: FLUX returned no images (check worker logs for FLUX AI Error / RuntimeError). HF space may have changed API or returned errors.")
            raise RuntimeError(
                "FLUX image generation produced no images. Check worker logs for 'FLUX AI Error' (e.g. RuntimeError from HF space). "
                "Ensure HF_API_KEY is valid and the space black-forest-labs/FLUX.1-schnell is up."
            )
        return True
    except Exception as e:
        logger.error(f"Media Task Error: {str(e)}")
        raise

@shared_task
def generate_voiceovers_task(script_id):
    """Generate voiceovers for all scenes"""
    try:
        script = VideoScript.objects.get(id=script_id)
        scenes = script.scenes.all().order_by('order')
        voice_gen = VoiceGenerator()
        voice_id = script.voice_id # Gemini voices: Puck, Charon, etc.
        
        # Cleanup old voiceovers
        AudioAsset.objects.filter(scene__script=script).delete()
        
        for scene in scenes:
            audio_file = voice_gen.generate_voiceover(scene.text, voice_id=voice_id)
            time.sleep(4)
            if not audio_file:
                continue
                
            with open(audio_file, 'rb') as f:
                audio_asset = AudioAsset.objects.create(scene=scene, voice_id=voice_id)
                # save() handles the actual file placement in settings.MEDIA_ROOT
                audio_asset.file.save(f'vo_{scene.id}.mp3', File(f))
                
            if os.path.exists(audio_file):
                os.unlink(audio_file)
        return True
    except Exception as e:
        logger.error(f"Voiceover Error: {str(e)}")
        raise

@shared_task(bind=True, max_retries=5)
def render_video_task(self, project_id):
    """Renders final video using FFmpeg inside the Docker container"""
    try:
        project = VideoProject.objects.get(id=project_id)
        # Use select_related/prefetch_related for cloud DB efficiency
        scenes = Scene.objects.filter(script__project=project).order_by('order').prefetch_related('media_assets')
        
        scenes_data = []
        missing_assets = []

        for scene in scenes:
            media = scene.media_assets.first()
            # Correcting the reverse lookup for AudioAsset (assumes OneToOne or ForeignKey)
            audio = getattr(scene, 'audio_asset', None)

            if not media or not os.path.exists(media.file.path):
                missing_assets.append(f"Scene {scene.order} Image")
            elif not audio or not os.path.exists(audio.file.path):
                missing_assets.append(f"Scene {scene.order} Audio")
            else:
                scenes_data.append({
                    'media_path': media.file.path,
                    'audio_path': audio.file.path,
                    'text': scene.text
                })

        # Retry if assets are missing (e.g. storage hasn't synced yet)
        if missing_assets:
            if self.request.retries >= self.max_retries:
                raise Exception(f"Final Render Attempt Failed: Missing {missing_assets}")
            logger.info(f"Retrying render... missing: {missing_assets}")
            raise self.retry(countdown=30)

        # Output folder for the render process
        render_output = os.path.join(settings.MEDIA_ROOT, 'renders', f"{uuid.uuid4().hex}.mp4")
        os.makedirs(os.path.dirname(render_output), exist_ok=True)

        editor = VideoEditor()
        # Ensure combine_videos is the correct method name in your VideoEditor class
        success = editor.combine_videos(project_id, scenes_data, render_output)
        
        if success:
            with open(render_output, 'rb') as f:
                RenderedVideo.objects.create(
                    project=project,
                    file=File(f, name=f"video_{project_id}.mp4")
                )
            project.status = 'completed'
            project.save()
            
            if os.path.exists(render_output):
                os.remove(render_output)
            return True
        else:
            raise Exception("FFmpeg process returned False")

    except Exception as e:
        logger.error(f"Render Error project {project_id}: {str(e)}")
        if self.request.retries >= self.max_retries:
            project.status = 'failed'
            project.save()
            # Only email on absolute failure
            send_mail(
                "Video Generation Failed",
                f"Project {project_id} failed. Error: {str(e)}",
                settings.DEFAULT_FROM_EMAIL,
                [project.user.email],
                fail_silently=True
            )
        raise self.retry(exc=e, countdown=60)
