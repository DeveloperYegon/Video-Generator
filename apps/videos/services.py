import os
import json
import tempfile
import subprocess
import requests
import nltk
import boto3
import google.generativeai as genai
from pexels_api import API
from django.conf import settings
from io import BytesIO
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk import BigramCollocationFinder, BigramAssocMeasures
import string

# Configure NLTK data path
nltk.data.path.append(os.path.join(settings.BASE_DIR, 'nltk_data'))

class ScriptGenerator:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_script(self, prompt, length=300):
        try:
            response = self.model.generate_content(
                f"Generate YouTube video script about: {prompt}\n"
                f"Length: {length} words\n"
                "Format: Narration with scene descriptions"
            )
            return response.text
        except Exception as e:
            return f"Script generation error: {str(e)}"

class ScriptProcessor:
    @staticmethod
    def segment_script(text):
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        scenes = []
        current_scene = []
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            current_scene.extend(sentences)
            if len(current_scene) >= 3:
                scenes.append(' '.join(current_scene))
                current_scene = []
        
        return scenes

    @staticmethod
    def extract_keywords(text, num_keywords=3):
        tokens = nltk.word_tokenize(text.lower())
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Add bigram detection
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(filtered_tokens)
        bigrams = finder.nbest(bigram_measures.pmi, 2)
        
        fdist = FreqDist(filtered_tokens)
        keywords = [word for word, _ in fdist.most_common(num_keywords * 2)]
        keywords += [' '.join(bigram) for bigram in bigrams]
        
        return sorted(list(set(keywords)))[:num_keywords]



class MediaFinder:
    def __init__(self):
        self.headers = {
            "Authorization": settings.PEXELS_API_KEY
        }

    def find_media(self, keywords, media_type='video'):
        try:
            query = ' '.join(keywords)
            base_url = "https://api.pexels.com/videos/search" if media_type == 'video' else "https://api.pexels.com/v1/search"
            params = {"query": query, "per_page": 1}

            response = requests.get(base_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if media_type == 'video' and data.get("videos"):
                video = data["videos"][0]
                return {
                    "url": video.get("url"),
                    "download_url": video["video_files"][0].get("link"),
                    "duration": video.get("duration")
                }
            elif media_type == 'photo' and data.get("photos"):
                photo = data["photos"][0]
                return {
                    "url": photo.get("url"),
                    "download_url": photo["src"].get("original"),
                    "width": photo.get("width"),
                    "height": photo.get("height")
                }

            return None
        except Exception as e:
            print(f"Pexels API Error: {str(e)}")
            return None

class VoiceGenerator:
    def __init__(self):
        self.client = boto3.client(
            'polly',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
    
    def generate_voiceover(self, text, voice_id='Joanna'):
        try:
            response = self.client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp.write(response['AudioStream'].read())
                return tmp.name
        except Exception as e:
            print(f"Voice generation error: {str(e)}")
            return None

class VideoEditor:
    @staticmethod
    def combine_scenes(scenes_data, output_path, project_id=None):
        """
        Combine multiple video and audio scenes into a single video
        
        Args:
            scenes_data: List of dictionaries with media_path and audio_path keys
            output_path: Path where the rendered video should be saved
            project_id: Optional VideoProject ID for organizing output
        
        Returns:
            Tuple of (success, output_file_path)
        """
        # Ensure we're using the media folder from Django settings
        if project_id:
            # Create project-specific subdirectory in media folder
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos', f'project_{project_id}')
        else:
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # If output_path is just a filename, append it to the output_dir
        if os.path.dirname(output_path) == '':
            output_path = os.path.join(output_dir, output_path)
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get absolute path for output
        output_path = os.path.abspath(output_path)
        
        # Standardize parameters
        TARGET_RESOLUTION = '1920x1080'
        TARGET_FRAMERATE = 30
        TARGET_AUDIO_SAMPLE_RATE = 44100
        TARGET_PIXEL_FORMAT = 'yuv420p'
        
        # Create temporary file for filter complex script
        filter_complex = []
        
        # Build input file arguments
        input_args = []
        for i, scene in enumerate(scenes_data):
            # Check if files exist
            if not os.path.exists(scene['media_path']):
                print(f"Media file not found: {scene['media_path']}")
                return False, None
            if not os.path.exists(scene['audio_path']):
                print(f"Audio file not found: {scene['audio_path']}")
                return False, None
                
            # Add files to input args
            input_args.extend(['-i', scene['media_path']])
            input_args.extend(['-i', scene['audio_path']])
        
        # Process each input pair (video, audio)
        video_streams = []
        audio_streams = []
        
        for i in range(len(scenes_data)):
            # Video input is at index 2*i, audio input at 2*i+1
            video_idx = 2*i
            audio_idx = 2*i+1
            
            # Video processing with detailed error handling
            video_streams.append(f"v{i}")
            filter_complex.append(
                f"[{video_idx}:v:0]scale={TARGET_RESOLUTION.split('x')[0]}:{TARGET_RESOLUTION.split('x')[1]}:force_original_aspect_ratio=decrease,"
                f"pad={TARGET_RESOLUTION.split('x')[0]}:{TARGET_RESOLUTION.split('x')[1]}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fps={TARGET_FRAMERATE},format={TARGET_PIXEL_FORMAT}[v{i}]"
            )
            
            # Audio processing with robust handling
            audio_streams.append(f"a{i}")
            filter_complex.append(
                f"[{audio_idx}:a:0]aformat=sample_rates={TARGET_AUDIO_SAMPLE_RATE}:"
                f"channel_layouts=stereo,volume=1.0,aresample={TARGET_AUDIO_SAMPLE_RATE}:async=1000[a{i}]"
            )
        
        # Build concatenation filter
        video_inputs = ''.join(f"[{v}]" for v in video_streams)
        audio_inputs = ''.join(f"[{a}]" for a in audio_streams)
        
        filter_complex.append(
            f"{video_inputs}concat=n={len(scenes_data)}:v=1:a=0[outv];"
            f"{audio_inputs}concat=n={len(scenes_data)}:v=0:a=1[outa]"
        )
        
        # Build final ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            *input_args,
            '-filter_complex', ';'.join(filter_complex),
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"FFmpeg output: {result.stdout}")
            
            # Verify the output file was created
            if os.path.exists(output_path):
                print(f"Successfully created video at: {output_path}")
                # Save to Django's RenderedVideo model
                relative_path = VideoEditor.get_relative_media_path(output_path)
                
                # Create RenderedVideo entry if project_id is provided
                if project_id:
                    try:
                        from apps.videos.models import VideoProject, RenderedVideo
                        
                        # Get video information for duration and resolution
                        video_info = VideoEditor.verify_media_file(output_path)
                        
                        # Extract duration and resolution
                        duration = 0
                        resolution = TARGET_RESOLUTION  # Default
                        
                        if video_info and 'streams' in video_info:
                            for stream in video_info['streams']:
                                if stream.get('codec_type') == 'video':
                                    # Calculate duration
                                    if 'duration' in stream:
                                        duration = float(stream['duration'])
                                    # Get resolution
                                    width = stream.get('width', 0)
                                    height = stream.get('height', 0)
                                    if width and height:
                                        resolution = f"{width}x{height}"
                                    break
                        
                        # Get or create the rendered video associated with this project
                        project = VideoProject.objects.get(id=project_id)
                        
                        # Delete existing rendered video if it exists
                        RenderedVideo.objects.filter(project=project).delete()
                        
                        # Create new rendered video
                        rendered_video = RenderedVideo.objects.create(
                            project=project,
                            file=relative_path,
                            duration_seconds=duration,
                            resolution=resolution
                        )
                        
                        # Update project status
                        project.status = 'completed'
                        project.save()
                        
                        print(f"Created RenderedVideo entry for project {project_id}")
                        
                    except Exception as e:
                        print(f"Error saving rendered video to database: {str(e)}")
                        # Continue anyway since the file was created successfully
                
                return True, relative_path
            else:
                print(f"Error: Output file not created at {output_path}")
                
                # Update project status if project_id was provided
                if project_id:
                    try:
                        from apps.videos.models import VideoProject
                        project = VideoProject.objects.get(id=project_id)
                        project.status = 'failed'
                        project.save()
                    except Exception as e:
                        print(f"Error updating project status: {str(e)}")
                
                return False, None
                
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            
            # Update project status if project_id was provided
            if project_id:
                try:
                    from apps.videos.models import VideoProject
                    project = VideoProject.objects.get(id=project_id)
                    project.status = 'failed'
                    project.save()
                except Exception as db_error:
                    print(f"Error updating project status: {str(db_error)}")
            
            return False, None
    
    @staticmethod
    def get_relative_media_path(absolute_path):
        """
        Convert an absolute file path to a relative path for Django media URLs
        
        Returns a path relative to MEDIA_ROOT for use in FileField
        """
        # If using Django's media root, return path relative to MEDIA_ROOT
        if settings.MEDIA_ROOT in absolute_path:
            return os.path.relpath(absolute_path, settings.MEDIA_ROOT)
            
        # Extract the part after 'media/'
        if 'media/' in absolute_path:
            parts = absolute_path.split('media/')
            if len(parts) > 1:
                return parts[1]
        
        # If not in expected format, just return the filename
        return os.path.basename(absolute_path)
            
    @staticmethod
    def verify_media_file(file_path):
        """Verify media file and return stream information"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_streams', '-print_format', 'json', file_path]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"FFprobe error for {file_path}: {e.stderr}")
            return None
            
    @staticmethod
    def process_single_video(input_path, output_path, project_id=None):
        """Process a single video to ensure it meets target specifications"""
        # Use Django's media folder structure
        if project_id:
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos', f'project_{project_id}')
        else:
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # If output_path is just a filename, append it to the output_dir
        if os.path.dirname(output_path) == '':
            output_path = os.path.join(output_dir, output_path)
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Target specs
        TARGET_RESOLUTION = '1920x1080'
        TARGET_FRAMERATE = 30
        TARGET_AUDIO_SAMPLE_RATE = 44100
        TARGET_PIXEL_FORMAT = 'yuv420p'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', f"scale={TARGET_RESOLUTION.split('x')[0]}:{TARGET_RESOLUTION.split('x')[1]}:force_original_aspect_ratio=decrease,"
                  f"pad={TARGET_RESOLUTION.split('x')[0]}:{TARGET_RESOLUTION.split('x')[1]}:(ow-iw)/2:(oh-ih)/2:color=black,"
                  f"fps={TARGET_FRAMERATE},format={TARGET_PIXEL_FORMAT}",
            '-af', f"aformat=sample_rates={TARGET_AUDIO_SAMPLE_RATE}:channel_layouts=stereo,"
                  f"aresample={TARGET_AUDIO_SAMPLE_RATE}:async=1000",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # If we have a project_id, update the RenderedVideo model
            if project_id:
                try:
                    from apps.videos.models import VideoProject, RenderedVideo
                    
                    # Get video information
                    video_info = VideoEditor.verify_media_file(output_path)
                    
                    # Extract duration and resolution
                    duration = 0
                    resolution = TARGET_RESOLUTION  # Default
                    
                    if video_info and 'streams' in video_info:
                        for stream in video_info['streams']:
                            if stream.get('codec_type') == 'video':
                                if 'duration' in stream:
                                    duration = float(stream['duration'])
                                width = stream.get('width', 0)
                                height = stream.get('height', 0)
                                if width and height:
                                    resolution = f"{width}x{height}"
                                break
                    
                    # Get or create the rendered video
                    project = VideoProject.objects.get(id=project_id)
                    
                    # Delete existing rendered video if it exists
                    RenderedVideo.objects.filter(project=project).delete()
                    
                    # Get path relative to MEDIA_ROOT
                    relative_path = VideoEditor.get_relative_media_path(output_path)
                    
                    # Create new rendered video
                    rendered_video = RenderedVideo.objects.create(
                        project=project,
                        file=relative_path,
                        duration_seconds=duration,
                        resolution=resolution
                    )
                    
                    # Update project status
                    project.status = 'completed'
                    project.save()
                    
                except Exception as e:
                    print(f"Error saving rendered video to database: {str(e)}")
            
            # Convert to relative path for Django media
            relative_path = VideoEditor.get_relative_media_path(output_path)
            return True, relative_path
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            
            # Update project status if project_id was provided
            if project_id:
                try:
                    from apps.videos.models import VideoProject
                    project = VideoProject.objects.get(id=project_id)
                    project.status = 'failed'
                    project.save()
                except Exception as db_error:
                    print(f"Error updating project status: {str(db_error)}")
                    
            return False, None