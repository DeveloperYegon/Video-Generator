# apps/videos/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages 
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse
from .models import VideoProject, VideoScript, Scene, MediaAsset, AudioAsset, RenderedVideo
from .services import ScriptGenerator, ScriptProcessor, MediaFinder, VoiceGenerator, VideoEditor
from .forms import VideoProjectForm, ScriptForm
import os
import requests
from django.conf import settings
from django.core.files import File
from tempfile import NamedTemporaryFile, mkstemp
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
    voice_gen = VoiceGenerator()
    voice_choices = [(voice_id, f"{voice_id} - {description}") for voice_id, description in voice_gen.get_available_voices().items()]

    try:
        script = project.script
        voice_id = script.voice_id
    except VideoScript.DoesNotExist:
        script = None
        voice_id = 'Matthew'  # Default voice

    if request.method == 'POST':
        form = ScriptForm(request.POST, voice_choices=voice_choices)
        if form.is_valid():
            text = form.cleaned_data['text']
            selected_voice_id = form.cleaned_data['voice_id']
            
            if script:
                script.original_text = text
                script.voice_id = selected_voice_id
                script.save()
            else:
                script = VideoScript.objects.create(
                    project=project,
                    original_text=text,
                    voice_id=selected_voice_id,
                    generated_by_ai=False
                )
            
            messages.success(request, "Script saved successfully.")
            
            # Check if the user clicked the Process button
            if 'process_script' in request.POST:
                return redirect('videos:process_script', project_id=project.id)
            return redirect('videos:edit_script', project_id=project.id)
    else:
        initial = {
            'text': script.original_text if script else '',
            'voice_id': voice_id
        }
        form = ScriptForm(initial=initial, voice_choices=voice_choices)

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
        voice_id = request.POST.get('voice_id', 'Matthew')

        generator = ScriptGenerator()
        generated_text = generator.generate_script(topic, length)

        # Save or update script
        try:
            script = project.script
            script.original_text = generated_text
            script.voice_id = voice_id
            script.generated_by_ai = True
            script.save()
        except VideoScript.DoesNotExist:
            script = VideoScript.objects.create(
                project=project,
                original_text=generated_text,
                voice_id=voice_id,
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
        return redirect('videos:edit_script', project_id=project.id)


@login_required
def find_media(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)
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
                
        except Exception as e:
            print(f"Error generating media for scene {scene.id}: {str(e)}")
            continue
    
    return redirect('videos:generate_voiceovers', project_id=project.id)
    
@login_required
def generate_voiceovers(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)
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
        
        # Create temporary output path
        _, temp_output = mkstemp(suffix='.mp4')
        
        # Render the video
        success, media_url = VideoEditor.combine_scenes(
            scenes_data=scenes_data, 
            output_path=temp_output,
            project_id=project_id
        )
        
        if success:
            messages.success(request, "Video rendered successfully!")
            # Clean up temporary file
            try:
                os.remove(temp_output)
            except OSError:
                pass
        else:
            messages.error(request, "Error rendering video. Check logs for details.")
        
        return redirect('videos:view_project', project_id=project_id)
    
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

@login_required
def delete_project(request, project_id):
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    
    if request.method == 'POST':
        # Delete associated files
        try:
            # Delete rendered video files
            if hasattr(project, 'rendered_video') and project.rendered_video:
                if project.rendered_video.file and os.path.exists(project.rendered_video.file.path):
                    os.remove(project.rendered_video.file.path)
            
            # Delete script-related files (media and audio)
            if hasattr(project, 'script'):
                for scene in project.script.scenes.all():
                    # Delete media assets
                    for media in scene.media_assets.all():
                        if media.file and os.path.exists(media.file.path):
                            os.remove(media.file.path)
                    
                    # Delete audio assets
                    if hasattr(scene, 'audio_asset') and scene.audio_asset:
                        if scene.audio_asset.file and os.path.exists(scene.audio_asset.file.path):
                            os.remove(scene.audio_asset.file.path)
        except Exception as e:
            print(f"Error deleting files: {str(e)}")
        
        # Delete project from database
        project.delete()
        messages.success(request, f"Project '{project.title}' has been deleted.")
        
        return redirect('videos:project_list')
    
    return render(request, 'videos/confirm_delete.html', {'project': project})