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
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
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
        self.api = API(settings.PEXELS_API_KEY)

    def find_media(self, keywords, media_type='video'):
        try:
            query = ' '.join(keywords)

            if media_type == 'video':
                self.api.search_videos(query, results_per_page=1)
                videos = self.api.get_entries()
                if videos:
                    video = videos[0]
                    return {
                        'url': video.video_url,
                        'download_url': video.video_files[0].link,
                        'duration': video.duration
                    }
            else:
                self.api.search(query, results_per_page=1)
                photos = self.api.get_entries()
                if photos:
                    photo = photos[0]
                    return {
                        'url': photo.url,
                        'download_url': photo.original,
                        'width': photo.width,
                        'height': photo.height
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
    def combine_scenes(scenes_data, output_path):
        try:
            inputs = []
            filters = []
            
            for i, scene in enumerate(scenes_data):
                inputs.extend(['-i', scene['media_path'], '-i', scene['audio_path']])
                
                if i > 0:
                    filters.append(
                        f'[{i}:v][{i}:a][{i+1}:v][{i+1}:a]'
                        f'concat=n=2:v=1:a=1[outv{i}][outa{i}]'
                    )
            
            filter_complex = ';'.join(filters)
            
            cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ]
            
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            return False