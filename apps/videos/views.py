
# apps/videos/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages 
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse
from .models import VideoProject, VideoScript, Scene, MediaAsset, AudioAsset, RenderedVideo
from .forms import VideoProjectForm, ScriptForm
from .tasks import process_video_project, process_script_task, find_media_task, generate_voiceovers_task, render_video_task
import os
import uuid
import logging

logger = logging.getLogger(__name__)

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
    
    from .services import VoiceGenerator
    voice_gen = VoiceGenerator()
    voice_choices = [(v, v) for v in voice_gen.get_available_voices()]
    
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

        from .services import ScriptGenerator
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
def processing_status(request, project_id, step):
    """
    Show processing status page with appropriate messaging based on processing step

    Args:
        step: One of 'script', 'media', 'voiceover', 'render'
    """
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)

    # If user hits refresh, check if we can proceed to next step
    if request.GET.get('check_status') == 'true':
        script = project.script

        if step == 'script':
            # Check if script processing is complete
            if script.scenes.exists():
                messages.success(request, "Script processing complete. Proceeding to media generation.")
                return redirect('videos:find_media', project_id=project.id)

        elif step == 'media':
            # Check if media processing is complete
            scenes = script.scenes.all()
            media_complete = True

            for scene in scenes:
                if not scene.media_assets.exists():
                    media_complete = False
                    break

            if media_complete and scenes.exists():
                messages.success(request, "Media generation complete. Proceeding to voiceover generation.")
                return redirect('videos:generate_voiceovers', project_id=project.id)

        elif step == 'voiceover':
            # Check if voiceover processing is complete
            scenes = script.scenes.all()
            audio_complete = True

            for scene in scenes:
                if not hasattr(scene, 'audio_asset') or not scene.audio_asset:
                    audio_complete = False
                    break

            if audio_complete and scenes.exists():
                messages.success(request, "Voiceover generation complete. Proceeding to video rendering.")
                return redirect('videos:render_video', project_id=project.id)

        elif step == 'render':
            # Check if rendering is complete
            if project.status == 'completed' and hasattr(project, 'rendered_video'):
                messages.success(request, "Video rendering complete!")
                return redirect('videos:view_project', project_id=project.id)

    # Set info messages based on step
    step_info = {
        'script': {
            'title': 'Processing Script',
            'message': 'We\'re analyzing your script and breaking it into scenes.',
            'next_step': 'Media Generation',
            'progress': 25
        },
        'media': {
            'title': 'Generating Media',
            'message': 'We\'re generating visual media for each scene in your video.',
            'next_step': 'Voiceover Generation',
            'progress': 50
        },
        'voiceover': {
            'title': 'Creating Voiceovers',
            'message': 'We\'re generating audio voiceovers for each scene in your video.',
            'next_step': 'Video Rendering',
            'progress': 75
        },
        'render': {
            'title': 'Rendering Video',
            'message': 'We\'re combining all elements to create your final video.',
            'next_step': 'Video Preview',
            'progress': 90
        }
    }

    context = {
        'project': project,
        'info': step_info.get(step, step_info['script']),
        'step': step,
        'refresh_url': reverse('videos:processing_status', args=[project_id, step]) + '?check_status=true'
    }

    return render(request, 'videos/processing_status.html', context)



    
@ login_required
def process_script(request, project_id):
    """Process script into scenes using Celery task"""
    try:
        project = get_object_or_404(VideoProject, id=project_id, user=request.user)
        script = get_object_or_404(VideoScript, project=project)

        # Update project status
        project.status = 'processing'
        project.save()

        # Run just the script processing task
        process_script_task.delay(script.id)

        messages.success(request, "Script processing started. Please wait for it to complete.")
        return redirect('videos:processing_status', project_id=project.id, step='script')
    except Exception as e:
        logger.error(f"Error starting script processing: {str(e)}")
        messages.error(request, f"Script processing failed: {str(e)}")
        return redirect('videos:edit_script', project_id=project.id)



@login_required
def find_media(request, project_id):
    """Find media for scenes using Celery task"""
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)

    # Check if scenes exist
    if not script.scenes.exists():
        messages.warning(request, "Scenes are still being processed. Please wait a few seconds and refresh.")
        return redirect('videos:edit_script', project_id=project.id)

    # Start media finding task
    find_media_task.delay(script.id)

    messages.success(request, "Media generation started. Please continue to the next step.")
    return redirect('videos:generate_voiceovers', project_id=project.id)


@login_required
def generate_voiceovers(request, project_id):
    """Generate voiceovers using Celery task"""
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    script = get_object_or_404(VideoScript, project=project)
    
    # Check if scenes exist
    if not script.scenes.exists():
        messages.warning(request, "No scenes found. Processing script first...")
        return redirect('videos:process_script', project_id=project.id)
    
    # Start voiceover generation task
    generate_voiceovers_task.delay(script.id)
    
    messages.success(request, "Voiceover generation started. Please continue to the final step.")
    return redirect('videos:render_video', project_id=project.id)


@login_required
def render_video(request, project_id):
    """Render video using Celery task"""
    try:
        project = VideoProject.objects.get(id=project_id, user=request.user)
        
        # Update project status to processing
        project.status = 'processing'
        project.save()
        
        # Start the rendering task
        render_video_task.delay(project_id)
        
        messages.success(request, "Video rendering started. You will be notified when it's complete.")
        return redirect('videos:view_project', project_id=project_id)
    
    except VideoProject.DoesNotExist:
        messages.error(request, "Project not found")
        return redirect('videos:project_list')
    except Exception as e:
        logger.error(f"Error starting video rendering: {str(e)}")
        messages.error(request, f"Error: {str(e)}")
        return redirect('videos:view_project', project_id=project.id)


@login_required
def process_full_video(request, project_id):
    """Process the entire video project in one go using Celery"""
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    
    # Check if script exists
    if not hasattr(project, 'script'):
        messages.error(request, "Please create a script first")
        return redirect('videos:edit_script', project_id=project.id)
    
    # Start the full video processing task
    process_video_project.delay(project_id)
    
    # Update project status
    project.status = 'queued'
    project.save()
    
    messages.success(request, "Video processing started. You'll be notified when it's complete.")
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
            logger.error(f"Error deleting files: {str(e)}")
        
        # Delete project from database
        project.delete()
        messages.success(request, f"Project '{project.title}' has been deleted.")
        
        return redirect('videos:project_list')
    
    return render(request, 'videos/confirm_delete.html', {'project': project})


@login_required
def retry_rendering(request, project_id):
    """Retry rendering a failed video"""
    project = get_object_or_404(VideoProject, id=project_id, user=request.user)
    
    if project.status != 'failed':
        messages.warning(request, "Only failed projects can be retried")
        return redirect('videos:view_project', project_id=project_id)
    
    # Reset status and start rendering
    project.status = 'queued'
    project.error_message = ''
    project.save()
    
    # Start the rendering task
    render_video_task.delay(project_id)
    
    messages.success(request, "Video rendering restarted. You will be notified when it's complete.")
    return redirect('videos:view_project', project_id=project_id)
