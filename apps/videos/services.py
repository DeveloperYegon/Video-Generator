# apps/videos/services.py
import nltk
import os
import json
import tempfile
import subprocess
import traceback
# import requests
import re
import boto3
import google.generativeai as genai
from gradio_client import Client
from gradio_client.exceptions import AppError
from django.conf import settings
import time
#from io import BytesIO
#from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk import BigramCollocationFinder, BigramAssocMeasures, word_tokenize
import string
import logging


logger = logging.getLogger(__name__)
# Configure NLTK data path
nltk.data.path.append(os.path.join(settings.BASE_DIR, 'nltk_data'))

class ScriptGenerator:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

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
    def __init__(self):
        # Ensure NLTK data is available in the Docker-mapped path
        self._prepare_nltk_data()

    def _prepare_nltk_data(self):
        """Ensures required NLTK resources are available locally"""
        required_resources = [
            'punkt', 
            'punkt_tab',
            'stopwords', 
            'averaged_perceptron_tagger'
        ]
        for resource in required_resources:
            try:
                nltk.data.find(f"tokenizers/{resource}" if 'punkt' in resource else f"corpora/{resource}")
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, download_dir=settings.NLTK_DATA_PATH, quiet=True)

    @staticmethod
    def segment_script(text):
        """Segments text into logical scenes (approx 3 sentences each)"""
        if not text:
            return []
            
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        scenes = []
        current_scene = []

        for para in paragraphs:
            # sent_tokenize requires 'punkt'
            sentences = sent_tokenize(para)
            for sent in sentences:
                current_scene.append(sent)
                # Grouping logic: adjust this number to change scene length
                if len(current_scene) >= 3:
                    scenes.append(' '.join(current_scene))
                    current_scene = []

        # Catch any remaining sentences
        if current_scene:
            scenes.append(' '.join(current_scene))

        return scenes

    @staticmethod
    def extract_keywords(text, num_keywords=3):
        """Extracts keywords and bigrams for AI Image Generation prompts"""
        try:
            tokens = word_tokenize(text.lower())
            # stopwords requires 'stopwords'
            stop_words = set(stopwords.words('english') + list(string.punctuation))
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

            if not filtered_tokens:
                return ["cinematic", "high quality"]

            # Bigram detection for better image prompts (e.g., "mountain bike")
            bigram_measures = BigramAssocMeasures()
            finder = BigramCollocationFinder.from_words(filtered_tokens)
            bigrams = finder.nbest(bigram_measures.pmi, 2)

            fdist = FreqDist(filtered_tokens)
            keywords = [word for word, _ in fdist.most_common(num_keywords)]
            keywords += [' '.join(bigram) for bigram in bigrams]

            # Standard style tags to improve FLUX results
            style_tags = ["4k", "cinematic lighting", "detailed"]
            
            final_keywords = list(set(keywords + style_tags))
            return final_keywords[:num_keywords + 2]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return ["cinematic", "visual", "atmospheric"]
            
class MediaFinder:
    """Finds/generates scene images via Hugging Face FLUX.1 Gradio. Lazy-inits client to avoid DNS/network errors at import."""

    FLUX_SPACE = "black-forest-labs/FLUX.1-schnell"

    def __init__(self):
        self.hf_token = getattr(settings, 'HF_API_KEY', os.getenv('HF_API_KEY'))
        self._client = None  # Lazy init so Docker worker doesn't fail on HF DNS in __init__

    def _get_client(self):
        """Create Gradio client on first use. Catches DNS/network errors (e.g. in Docker with no outbound DNS)."""
        if self._client is not None:
            return self._client
        try:
            self._client = Client(self.FLUX_SPACE, token=self.hf_token)
            return self._client
        except Exception as e:
            logger.exception("FLUX Gradio client init failed (check network/DNS and HF_API_KEY).")
            raise

    def generate_image(self, prompt, width=1024, height=1024):
        client = self._get_client()  # Let ConnectError/OSError propagate so task can raise a clear message
        max_attempts = 3
        delays = (5, 15)  # seconds between retries (transient OOM/queue on HF free tier)
        for attempt in range(1, max_attempts + 1):
            try:
                result = client.predict(
                    prompt=prompt,
                    seed=0,
                    randomize_seed=True,
                    width=width,
                    height=height,
                    num_inference_steps=4,
                    api_name="/infer"
                )
                # API returns (image_path, seed) per space config; extract path
                if isinstance(result, (list, tuple)):
                    result = next((x for x in result if isinstance(x, str) and ("/" in x or "\\" in x)), None)
                if result and isinstance(result, str) and os.path.exists(result):
                    return result
                logger.error(f"FLUX result not a valid file path: {type(result).__name__!r} = {result!r}")
                return None
            except AppError as e:
                # HF space often returns RuntimeError (OOM, queue timeout, or backend error) on free tier; retry
                if attempt < max_attempts:
                    wait = delays[attempt - 1] if attempt <= len(delays) else delays[-1]
                    logger.warning("FLUX AppError (attempt %s/%s), retry in %ss: %s", attempt, max_attempts, wait, e)
                    time.sleep(wait)
                else:
                    logger.exception("FLUX AI Error after %s attempts: %s", max_attempts, e)
                    return None
            except Exception as e:
                logger.exception("FLUX AI Error: %s (full: %r)", e, e)
                return None
        return None

class VoiceGenerator:
    def __init__(self):
        self.available_voices = {
            'Matthew': 'Male (US)', 'Joanna': 'Female (US)', 'Ruth': 'Female (US)',
            'Stephen': 'Male (US)', 'Amy': 'Female (British)', 'Brian': 'Male (British)'
        }
        try:
            self.client = boto3.client(
                'polly',
                region_name=getattr(settings, 'AWS_REGION', 'us-east-1'),
                aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
            )
        except Exception as e:
            logger.error(f"AWS Polly Init Failed: {e}")
            self.client = None 


    def get_available_voices(self):
        """Returns the dictionary of voices for the UI dropdown"""
        return self.available_voices

    def clean_narration_text(self, text):
        """Removes [Scene Descriptions] and (Parentheticals)"""
        if not text: return ""
        cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        cleaned_text = re.sub(r'(?i)^(scene|description):\s*', '', cleaned_text)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

    def generate_voiceover(self, text, voice_id='Matthew'):
        try:
            narration_text = self.clean_narration_text(text)
            if not narration_text: return None

            response = self.client.synthesize_speech(
                Text=narration_text[:2999],
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='neural'
            )

            # Use delete=False so tasks.py can move the file before it's destroyed
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp.write(response['AudioStream'].read())
                return tmp.name
        except Exception as e:
            logger.error(f"Polly Error: {str(e)}")
            return None

class VideoEditor:
    @staticmethod
    def combine_videos(project_id, scenes_data, output_path):
        logger.info(f"Rendering project {project_id} in Docker environment")
        
        if not scenes_data:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ffmpeg_cmd = ["ffmpeg", "-y", "-threads", "0"]

        for scene in scenes_data:
            # -loop 1 is vital for static images to act as video
            ffmpeg_cmd.extend(["-loop", "1", "-i", scene["media_path"]])
            ffmpeg_cmd.extend(["-i", scene["audio_path"]])

        filter_complex = []
        v_labels = []
        a_labels = []

        for i, scene in enumerate(scenes_data):
            v_in, a_in = 2*i, 2*i+1
            
            # Pad images to 1080p (Prevents crash if FLUX returns odd dimensions)
            filter_complex.append(
                f"[{v_in}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30[v{i}];"
            )
            filter_complex.append(f"[{a_in}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}];")
            
            v_labels.append(f"[v{i}]")
            a_labels.append(f"[a{i}]")

        concat_str = f"{''.join(v_labels)}{''.join(a_labels)}concat=n={len(scenes_data)}:v=1:a=1[outv][outa]"
        filter_complex.append(concat_str)

        ffmpeg_cmd.extend([
            "-filter_complex", "".join(filter_complex),
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", # Forces video to end exactly when audio ends
            output_path
        ])

        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg Render Error: {e.stderr[-500:]}")
            return False
