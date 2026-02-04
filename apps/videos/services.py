# apps/videos/services.py
import os
import json
import tempfile
import subprocess
import traceback
from venv import logger
#import requests
import nltk
import re
import boto3
import google.generativeai as genai
from gradio_client import Client  # MODIFIED: Added for FLUX.1
from django.conf import settings
#from io import BytesIO
#from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk import BigramCollocationFinder, BigramAssocMeasures
import string
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

        #return sorted(list(set(keywords)))[:num_keywords]
        return ["cinematic", "visual"]  # Simplified for brevity

class MediaFinder:
    def __init__(self):
        # MODIFIED: Initializing FLUX.1 [schnell] client with HF Token
        self.client = Client("black-forest-labs/FLUX.1-schnell", hf_token=settings.HF_API_KEY)

    def generate_image(self, prompt, width=1024, height=1024):
        """Generates an image using FLUX and returns the temporary local path"""
        try:
            # MODIFIED: Calling FLUX.1 API. Returns local path to temp image.
            result = self.client.predict(
                prompt=prompt,
                seed=0,
                randomize_seed=True,
                width=width,
                height=height,
                num_inference_steps=4,  # Schnell is optimized for 4 steps
                api_name="/infer"
            )
            return result  # This is a path string
        except Exception as e:
            logger.error(f"FLUX AI Error: {str(e)}")
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
                Text=narration_text[:2999],  # Polly limit,
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
    def combine_videos(project_id, scenes_data, output_path):
        """
        Combine multiple scenes into a single video with audio and text overlays.
        scenes_data: List of dicts with 'media_path', 'audio_path', 'text', 'duration_seconds'
        output_path: Path to save the rendered video
        """
        logger.info(f"Starting video combination for project_id={project_id}, output_path={output_path}")
        logger.info(f"Number of scenes to process: {len(scenes_data)}")

        if not scenes_data:
            logger.error("No scenes provided for video combination")
            return False

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build FFmpeg command
        ffmpeg_cmd = ["ffmpeg", "-y"]  # -y to overwrite output file if exists

        # Add input files (video and audio for each scene)
        input_files = []
        for scene in scenes_data:
            input_files.extend(["-i", scene["media_path"]])  # Video input
            input_files.extend(["-i", scene["audio_path"]])  # Audio input

        # Build filter_complex for video and audio processing
        filter_complex = []
        video_labels = []
        audio_labels = []

        for i, scene in enumerate(scenes_data):
            video_idx = 2 * i  # Video input index (0, 2, 4, 6 for 4 scenes)
            audio_idx = 2 * i + 1  # Audio input index (1, 3, 5, 7 for 4 scenes)

            # Video processing: scale to 1920x1080, set fps, adjust timestamps
            video_filter = (
                f"[{video_idx}:v]scale=1920:1080[v{i}_scaled];"
                f"[v{i}_scaled]fps=30[v{i}_fps];"
                f"[v{i}_fps]setpts=PTS-STARTPTS[v{i}]"
            )
            filter_complex.append(video_filter)
            video_labels.append(f"[v{i}]")

            # Audio processing: set format, resample, adjust timestamps
            audio_filter = (
                f"[{audio_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"aresample=async=1,asetpts=PTS-STARTPTS[a{i}]"
            )
            filter_complex.append(audio_filter)
            audio_labels.append(f"[a{i}]")

        # Concatenate video and audio streams
        filter_complex.append(
            f"{' '.join(video_labels)}concat=n={len(scenes_data)}:v=1:a=0[outv];"
            f"{' '.join(audio_labels)}concat=n={len(scenes_data)}:v=0:a=1[outa]"
        )

        # Join all filter chains
        filter_complex_str = "".join(filter_complex)

        # Add filter_complex and output mapping
        ffmpeg_cmd.extend([
            *input_files,
            "-filter_complex", filter_complex_str,
            "-map", "[outv]",  # Map video output
            "-map", "[outa]",  # Map audio output
            "-c:v", "libx264",  # Video codec
            "-preset", "medium",  # Encoding speed vs. compression
            "-c:a", "aac",  # Audio codec
            "-b:a", "192k",  # Audio bitrate
            "-r", "30",  # Frame rate
            "-s", "1920x1080",  # Resolution
            "-f", "mp4",  # Output format
            output_path
        ])

        try:
            # Run FFmpeg command
            logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"FFmpeg completed successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.returncode}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during video combination: {str(e)}")
            return False
    @staticmethod
    def _extract_ffmpeg_error(stderr_content):
        """Extract meaningful error message from FFmpeg stderr output"""
        # Common FFmpeg error patterns
        import re

        patterns = [
            r"No such file or directory: '(.*?)'",
            r"Invalid data found when processing input",
            r"Error while opening encoder for output stream",
            r"Error opening input files: (.*?)\n",
            r"Conversion failed!",
            r"Output file is empty",
            r"(.*?) not found",
            r"Unsupported codec.*?: (.*?)\n"
        ]

        for pattern in patterns:
            match = re.search(pattern, stderr_content)
            if match:
                return match.group(0)

        # If no specific pattern found, return the last line (usually contains the error)
        lines = stderr_content.strip().split('\n')
        if lines:
            return lines[-1]

        return "Unknown FFmpeg error"

    @staticmethod
    def _update_project_status(project_id, status, error_message=None):
        """Update project status and error message"""
        if not project_id:
            return

        try:
            from .models import VideoProject
            project = VideoProject.objects.get(id=project_id)
            project.status = status
            if error_message:
                project.error_message = error_message[:255]  # Truncate to fit field
            project.save()

            logger = logging.getLogger(__name__)
            logger.info(f"Updated project {project_id} status to '{status}'")

            # Notify user of failure if applicable
            if status == 'failed' and project.user.email:
                from django.core.mail import send_mail
                from django.conf import settings

                send_mail(
                    'Video processing failed',
                    f'Unfortunately, your video "{project.title}" could not be processed. Error: {error_message}',
                    settings.DEFAULT_FROM_EMAIL,
                    [project.user.email],
                    fail_silently=True,
                )

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error updating project status: {str(e)}")

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
        """
        Verify media file and return stream information

        Returns:
            dict: JSON response from ffprobe or None if verification fails
        """
        import logging
        logger = logging.getLogger(__name__)

        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return None

        try:
            # Create a temporary file for stderr output
            with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as stderr_file:
                stderr_path = stderr_file.name

            # Run ffprobe with a timeout
            cmd = ['ffprobe', '-v', 'error', '-show_streams', '-print_format', 'json', file_path]
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=open(stderr_path, 'w'),
                text=True,
                timeout=30  # Add timeout
            )

            # Parse and validate the JSON output
            info = json.loads(result.stdout)

            # Check if we have valid stream information
            if 'streams' not in info or len(info['streams']) == 0:
                with open(stderr_path, 'r') as f:
                    stderr_content = f.read()
                logger.error(f"No valid streams found in {file_path}")
                logger.error(f"FFprobe stderr: {stderr_content}")
                return None

            # Log basic media info
            for stream in info['streams']:
                codec_type = stream.get('codec_type', 'unknown')
                codec_name = stream.get('codec_name', 'unknown')
                if codec_type == 'video':
                    width = stream.get('width', 'unknown')
                    height = stream.get('height', 'unknown')
                    duration = stream.get('duration', 'unknown')
                    logger.debug(f"Video stream: {codec_name} {width}x{height}, duration: {duration}s")
                elif codec_type == 'audio':
                    sample_rate = stream.get('sample_rate', 'unknown')
                    channels = stream.get('channels', 'unknown')
                    logger.debug(f"Audio stream: {codec_name}, {sample_rate}Hz, {channels} channels")

            return info

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FFprobe output for {file_path}: {str(e)}")
            return None
        except subprocess.CalledProcessError as e:
            with open(stderr_path, 'r') as f:
                stderr_content = f.read()
            logger.error(f"FFprobe error for {file_path}: {stderr_content}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"FFprobe timed out for {file_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error verifying {file_path}: {str(e)}")
            return None
        finally:
            # Clean up stderr log file
            if 'stderr_path' in locals() and os.path.exists(stderr_path):
                try:
                    os.remove(stderr_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up stderr log file: {str(e)}")

    @staticmethod
    def process_single_video(input_path, output_path, project_id=None):
        """
        Process a single video to ensure it meets target specifications

        Args:
            input_path: Path to the input video
            output_path: Path where the output video will be saved
            project_id: Optional ID of the VideoProject to update status

        Returns:
            Tuple of (success: bool, media_url: str)
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Processing video: input={input_path}, output={output_path}, project_id={project_id}")

        # Validate input file
        if not os.path.exists(input_path):
            error_msg = f"Input file does not exist: {input_path}"
            logger.error(error_msg)
            VideoEditor._update_project_status(project_id, 'failed', error_msg)
            return False, None

        # Verify input video can be processed
        input_info = VideoEditor.verify_media_file(input_path)
        if not input_info:
            error_msg = f"Input file is invalid or corrupted: {input_path}"
            logger.error(error_msg)
            VideoEditor._update_project_status(project_id, 'failed', error_msg)
            return False, None

        # Use Django's media folder structure
        if project_id:
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos', f'project_{project_id}')
        else:
            output_dir = os.path.join(settings.MEDIA_ROOT, 'rendered_videos')

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            error_msg = f"Failed to create output directory: {str(e)}"
            logger.error(error_msg)
            VideoEditor._update_project_status(project_id, 'failed', error_msg)
            return False, None

        # If output_path is just a filename, append it to the output_dir
        if os.path.dirname(output_path) == '':
            output_path = os.path.join(output_dir, output_path)
        else:
            # Ensure the directory exists
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            except Exception as e:
                error_msg = f"Failed to create output directory: {str(e)}"
                logger.error(error_msg)
                VideoEditor._update_project_status(project_id, 'failed', error_msg)
                return False, None

        # Target specs
        TARGET_RESOLUTION = '1920x1080'
        TARGET_FRAMERATE = 30
        TARGET_AUDIO_SAMPLE_RATE = 44100
        TARGET_PIXEL_FORMAT = 'yuv420p'

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf',
            f"scale={TARGET_RESOLUTION.split('x')[0]}:{TARGET_RESOLUTION.split('x')[1]}:force_original_aspect_ratio=decrease,"
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

        logger.debug(f"FFmpeg processing command: {' '.join(cmd)}")

        try:
            # Create a temporary file for stderr output
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as stderr_file:
                stderr_path = stderr_file.name

            # Run the FFmpeg command with a timeout
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=open(stderr_path, 'w'),
                    text=True,
                    timeout=600  # 10 minute timeout for processing
                )

                # Verify the output was created successfully
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    with open(stderr_path, 'r') as f:
                        stderr_content = f.read()

                    error_msg = f"FFmpeg failed to create a valid output file (file missing or empty)"
                    logger.error(error_msg)
                    logger.error(f"FFmpeg stderr: {stderr_content}")
                    VideoEditor._update_project_status(project_id, 'failed', error_msg)
                    return False, None

                # Verify the output file is a valid video
                output_info = VideoEditor.verify_media_file(output_path)
                if not output_info:
                    error_msg = f"Output file verification failed: {output_path}"
                    logger.error(error_msg)
                    VideoEditor._update_project_status(project_id, 'failed', error_msg)
                    return False, None

                # If we have a project_id, update the RenderedVideo model
                if project_id:
                    try:
                        from .models import VideoProject, RenderedVideo

                        # Extract duration and resolution
                        duration = 0
                        resolution = TARGET_RESOLUTION  # Default

                        if output_info and 'streams' in output_info:
                            for stream in output_info['streams']:
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
                        project.error_message = ''  # Clear any error messages
                        project.save()

                        logger.info(f"Successfully processed video for project {project_id}")
                        logger.info(f"Duration: {duration}s, Resolution: {resolution}")

                    except Exception as e:
                        error_msg = f"Error saving rendered video to database: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        VideoEditor._update_project_status(project_id, 'failed', error_msg)
                        return False, None

                # Convert to relative path for Django media
                relative_path = VideoEditor.get_relative_media_path(output_path)
                return True, relative_path

            except subprocess.CalledProcessError as e:
                # Read stderr content from file
                with open(stderr_path, 'r') as f:
                    stderr_content = f.read()

                error_msg = f"FFmpeg processing error: {e.returncode}"
                logger.error(error_msg)
                logger.error(f"FFmpeg stderr: {stderr_content}")

                # Try to extract more specific error details from stderr
                specific_error = VideoEditor._extract_ffmpeg_error(stderr_content)
                if specific_error:
                    error_msg = f"FFmpeg error: {specific_error}"

                VideoEditor._update_project_status(project_id, 'failed', error_msg)
                return False, None

            except subprocess.TimeoutExpired:
                error_msg = "Video processing timed out (exceeded 10 minutes)"
                logger.error(error_msg)
                VideoEditor._update_project_status(project_id, 'failed', error_msg)
                return False, None

        except Exception as e:
            error_msg = f"General error during video processing: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            VideoEditor._update_project_status(project_id, 'failed', error_msg)
            return False, None
        finally:
            # Clean up stderr log file
            if 'stderr_path' in locals() and os.path.exists(stderr_path):
                try:
                    os.remove(stderr_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up stderr log file: {str(e)}")