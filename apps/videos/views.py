
#/apps/videos/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages 
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import VideoProject, VideoScript, Scene, MediaAsset, AudioAsset, RenderedVideo
from .services import ScriptGenerator, ScriptProcessor, MediaFinder, VoiceGenerator, VideoEditor
from .forms import VideoProjectForm, ScriptForm
import os
import requests
from django.conf import settings
from django.core.files import File
from tempfile import NamedTemporaryFile
import uuid


@login_required
def create_project(request):
    if request.method == 'POST':
        form = VideoProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()
            return redirect('videos:edit_script', project_id=project.id)
    else:
        form = VideoProjectForm()
    return render(request, 'videos/create_project.html', {'form': form})


@login_required
def edit_script(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)

    try:
        script = project.script
    except VideoScript.DoesNotExist:
        script = None

    if request.method == 'POST':
        form = ScriptForm(request.POST)
        if form.is_valid():
            if script:
                script.original_text = form.cleaned_data['text']
                script.save()
            else:
                script = VideoScript.objects.create(
                    project=project,
                    original_text=form.cleaned_data['text'],
                    generated_by_ai=False
                )
            return redirect('videos:rocess_script', project_id=project.id)
    else:
        initial = {'text': script.original_text if script else ''}
        form = ScriptForm(initial=initial)

    return render(request, 'videos/edit_script.html', {
        'project': project,
        'form': form,
        'script': script
    })


@login_required
def generate_script(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)

    if request.method == 'POST':
        topic = request.POST.get('topic', '')
        length = request.POST.get('length', 300)

        generator = ScriptGenerator()
        generated_text = generator.generate_script(topic, length)

        # Save or update script
        try:
            script = project.script
            script.original_text = generated_text
            script.generated_by_ai = True
            script.save()
        except VideoScript.DoesNotExist:
            script = VideoScript.objects.create(
                project=project,
                original_text=generated_text,
                generated_by_ai=True
            )

        return JsonResponse({'status': 'success', 'text': generated_text})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


@login_required
def process_script(request, project_id):
    try:
        project = get_object_or_404(VideoProject, id=project_id, user=request.user)
        script = get_object_or_404(VideoScript, project=project)

        # Segment script into scenes
        processor = ScriptProcessor()
        sentences = processor.segment_script(script.original_text)

        # Clear existing scenes
        Scene.objects.filter(script=script).delete()

        # Create new scenes
        scenes = []
        for i, sentence in enumerate(sentences):
            keywords = processor.extract_keywords(sentence)
            scene = Scene.objects.create(
                script=script,
                order=i + 1,
                text=sentence,
                keywords=keywords
            )
            scenes.append(scene)
              
        return redirect('videos:find_media', project_id=project.id)
    except Exception as e:
        messages.error(request, f"Script processing failed: {str(e)}")
        return redirect('edit_script', project_id=project.id)


@login_required
def find_media(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)
    scenes = script.scenes.all().order_by('order')

    media_finder = MediaFinder()

    for scene in scenes:
        # Skip if media already exists
        if scene.media_assets.exists():
            continue

        keywords = scene.keywords
        # First try to find a video
        media = media_finder.find_media(keywords, media_type='video')

        if not media:
            # Fall back to image if no video found
            media = media_finder.find_media(keywords, media_type='photo')
            if not media:
                continue

        # Download the media
        try:
            response = requests.get(media['download_url'], stream=True)
            if response.status_code == 200:
                ext = 'mp4' if 'video' in media['download_url'] else 'jpg'
                temp_file = NamedTemporaryFile(delete=False, suffix=f'.{ext}')

                for chunk in response.iter_content(1024):
                    temp_file.write(chunk)

                temp_file.close()

                # Create media asset
                media_asset = MediaAsset.objects.create(
                    scene=scene,
                    asset_type='video' if 'video' in media['download_url'] else 'image',
                    source='pexels',
                    url=media['url']
                )

                # Save the file
                with open(temp_file.name, 'rb') as f:
                    media_asset.file.save(f'{uuid.uuid4()}.{ext}', File(f))

                # Set duration (5 seconds for images, actual duration for videos)
                if media_asset.asset_type == 'image':
                    media_asset.duration_seconds = 5.0
                else:
                    media_asset.duration_seconds = media.get('duration', 5.0)

                media_asset.save()

                # Clean up
                os.unlink(temp_file.name)
        except Exception as e:
            print(f"Error downloading media: {e}")

    return redirect('videos:generate_voiceovers', project_id=project.id)


@login_required
def generate_voiceovers(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)
    scenes = script.scenes.all().order_by('order')

    voice_gen = VoiceGenerator()

    for scene in scenes:
        # Skip if audio already exists
        if hasattr(scene, 'audio_asset'):
            continue

        # Generate voiceover
        audio_file = voice_gen.generate_voiceover(scene.text)

        # Create audio asset
        with open(audio_file, 'rb') as f:
            audio_asset = AudioAsset.objects.create(
                scene=scene,
                voice_id='Joanna'  # Default voice
            )
            audio_asset.file.save(f'{uuid.uuid4()}.mp3', File(f))

        # Clean up
        os.unlink(audio_file)

    return redirect('videos:render_video', project_id=project.id)


@login_required
def render_video(request, project_id):
    """View for rendering the video"""
    try:
        project = VideoProject.objects.get(id=project_id, user=request.user)
        script = get_object_or_404(VideoScript, project=project)
        
        # Update project status to processing
        project.status = 'processing'
        project.save()
        
        # Get all scenes for this project
        scenes_data = []
        scenes = Scene.objects.filter(script=script).order_by('order')
        
        for scene in scenes:
            media_asset = scene.media_assets.first()
            audio_asset = getattr(scene, 'audio_asset', None)
            
            if media_asset and audio_asset:
                # Ensure file paths exist
                if not os.path.exists(media_asset.file.path):
                    messages.error(request, f"Media file not found: {media_asset.file.name}")
                    return redirect('videos:view_project', project_id=project.id)
                    
                if not os.path.exists(audio_asset.file.path):
                    messages.error(request, f"Audio file not found: {audio_asset.file.name}")
                    return redirect('videos:view_project', project_id=project.id)
                
                scenes_data.append({
                    'media_path': media_asset.file.path,
                    'audio_path': audio_asset.file.path,
                    'text': scene.text
                })
        
        if not scenes_data:
            messages.error(request, "No scenes found with both media and audio")
            project.status = 'failed'
            project.save()
            return redirect('videos:view_project', project_id=project.id)
        
        # Create the output filename
        output_filename = f"project_{project_id}_final.mp4"
        
        # Render the video - no need to manually create output directories anymore
        # The updated VideoEditor will handle this and create the RenderedVideo model entry
        success, media_url = VideoEditor.combine_scenes(
            scenes_data=scenes_data, 
            output_path=output_filename,
            project_id=project_id
        )
        
        if success:
            messages.success(request, "Video rendered successfully!")
        else:
            messages.error(request, "Error rendering video. Check logs for details.")
            # Project status will already be updated by the VideoEditor
            
        return redirect('videos:view_project', project_id=project.id)
    
    except VideoProject.DoesNotExist:
        messages.error(request, "Project not found")
        return redirect('videos:project_list')
    except Exception as e:
        import traceback
        print(f"Error rendering video: {str(e)}")
        print(traceback.format_exc())
        
        # Update project status to failed
        try:
            project.status = 'failed'
            project.save()
        except:
            pass  # If we can't update the project status, continue with the error response
            
        messages.error(request, f"Error: {str(e)}")
        return redirect('videos:view_project', project_id=project.id)

@login_required
def view_project(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    return render(request, 'videos/view_project.html', {'project': project})

@login_required
def project_list(request):
    projects = VideoProject.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'videos/project_list.html', {'projects': projects})