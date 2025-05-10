import os
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

REQUIRED_NLTK_RESOURCES = [
    'punkt',
    'stopwords',
    'punkt_tab',
    'wordnet',
    'averaged_perceptron_tagger'
]

def download_nltk_data():
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
    
    for resource in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.download(resource, download_dir=NLTK_DATA_PATH)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    # Add custom path to NLTK's data path
    nltk.data.path.append(NLTK_DATA_PATH)
    print(f"NLTK data path configured: {nltk.data.path}")

if __name__ == "__main__":
    download_nltk_data()