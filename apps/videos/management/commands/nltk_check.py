# apps/videos/management/commands/setup_nltk.py
import os
import nltk
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Downloads and configures required NLTK datasets'

    def handle(self, *args, **options):
        required_data = {
            'tokenizers/punkt': 'punkt',
            'tokenizers/punkt_tab': 'punkt_tab',  # Added this line
            'corpora/stopwords': 'stopwords',
            'corpora/wordnet': 'wordnet',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }

        nltk_data_path = os.path.join(settings.BASE_DIR, 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        for path, package in required_data.items():
            try:
                nltk.data.find(path)
                self.stdout.write(self.style.SUCCESS(f'✓ {package} already installed'))
            except LookupError:
                self.stdout.write(f'Downloading {package}...')
                nltk.download(package, download_dir=nltk_data_path)
                self.stdout.write(self.style.SUCCESS(f'✓ {package} installed'))

        self.stdout.write(self.style.SUCCESS('\nNLTK setup completed!'))
        self.stdout.write(f'Data location: {nltk_data_path}')