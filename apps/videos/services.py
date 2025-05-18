# apps/videos/services.py
import os
import json
import tempfile
import subprocess
import requests
import tempfile
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
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client
import logging

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
        self.stability_api = client.StabilityInference(
            key=settings.STABILITY_API_KEY,
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0"
        )

    def generate_image(self, prompt, width=1024, height=1024, steps=30):
        try:
            answers = self.stability_api.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=8.0,
                sampler=generation.SAMPLER_K_DPMPP_2M
            )
            
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        return artifact.binary
            return None
        except Exception as e:
            logging.error(f"Stability AI Error: {str(e)}")
            return None


class VoiceGenerator:
    def __init__(self):
        self.client = boto3.client(
            'polly',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        
        # Available voices from Amazon Polly (Neural voices)
        self.available_voices = {
            'Matthew': 'Male (US)',
            'Joanna': 'Female (US)',
            'Lupe': 'Female (US, Spanish)',
            'Ruth': 'Female (US)',
            'Stephen': 'Male (US)',
            'Olivia': 'Female (Australian)',
            'Amy': 'Female (British)',
            'Emma': 'Female (British)',
            'Brian': 'Male (British)',
            'Aria': 'Female (New Zealand)',
            'Gabrielle': 'Female (French)',
            'Vicki': 'Female (German)',
            'Takumi': 'Male (Japanese)',
            'Lucia': 'Female (Spanish)',
            'Camila': 'Female (Brazilian Portuguese)'
        }
    
    def get_available_voices(self):
        """Returns a dictionary of available voices and their descriptions"""
        return self.available_voices
    
    def clean_narration_text(self, text):
        """
        Clean the text to remove scene descriptions (text in brackets or parentheses)
        and any other non-narration elements
        """
        import re
        
        # Remove text between brackets [...] or parentheses (...)
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        
        # Remove any "Scene:" or "Description:" prefixes
        cleaned_text = re.sub(r'(?i)^(scene|description):\s*', '', cleaned_text)
        
        # Clean up multiple spaces and line breaks
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def generate_voiceover(self, text, voice_id='Matthew'):
        """Generate voiceover with specified voice ID"""
        try:
            # Clean the text to remove scene descriptions
            narration_text = self.clean_narration_text(text)
            
            # Skip empty text after cleaning
            if not narration_text.strip():
                return None
                
            response = self.client.synthesize_speech(
                Text=narration_text,
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        inputs = []
        filter_chains = []
        video_streams = []
        audio_streams = []
        
        # Standardization parameters
        TARGET_RESOLUTION = '1920x1080'
        TARGET_FRAMERATE = 30
        TARGET_AUDIO_SAMPLE_RATE = 44100
        
        for index, scene in enumerate(scenes_data):
            # Add inputs
            inputs.extend(['-i', scene['media_path']])
            inputs.extend(['-i', scene['audio_path']])
            
            # Video processing chain
            video_filter = [
                f"[{index*2}:v]scale={TARGET_RESOLUTION}[v{index}_scaled];",
                f"v{index}_scaled,fps={TARGET_FRAMERATE}[v{index}_fps];",
                f"v{index}_fps,setpts=PTS-STARTPTS[v{index}]"
            ]
            
            # Audio processing chain
            audio_filter = [
                f"[{index*2+1}:a]aformat=sample_rates={TARGET_AUDIO_SAMPLE_RATE}:channel_layouts=stereo,",
                "aresample=async=1,",
                "asetpts=PTS-STARTPTS,",
                f"arealtime[a{index}]"
            ]
            
            filter_chains.extend(video_filter)
            filter_chains.extend(audio_filter)
            
            video_streams.append(f"[v{index}]")
            audio_streams.append(f"[a{index}]")
        
        # Concatenation filter
        filter_chains.append(
            f"{''.join(video_streams)}concat=n={len(scenes_data)}:v=1:a=0[outv];"
            f"{''.join(audio_streams)}concat=n={len(scenes_data)}:v=0:a=1[outa]"
        )
        
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', ''.join(filter_chains),
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if project_id:
                # Use process_single_video to handle DB updates
                final_output = os.path.join(
                    settings.MEDIA_ROOT,
                    'rendered_videos',
                    f'project_{project_id}',
                    os.path.basename(output_path))
                
                success, media_url = VideoEditor.process_single_video(
                    output_path,
                    final_output,
                    project_id
                )
                return success, media_url
                
            return True, output_path
            
        except subprocess.CalledProcessError as e:
            if project_id:
                VideoEditor._update_project_status(project_id, 'failed')
            return False, None

    @staticmethod
    def _update_project_status(project_id, status):
        try:
            from apps.videos.models import VideoProject
            project = VideoProject.objects.get(id=project_id)
            project.status = status
            project.save()
        except Exception as e:
            print(f"Status update error: {str(e)}")
    
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