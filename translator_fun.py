import pickle
import nltk
import gzip
from pathlib import Path
import TranslaterLogic
import requests

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

APP_DIR = Path(__file__).parent.absolute()
# Fixed Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CaoGi4R0u4udNHmjBh1hsJdvkdDzN0dc"
MODEL_PATH = APP_DIR / "trigramTrans.gz"
_translator = None

def download_model():
    """Download the model file from Google Drive"""
    print(f"Downloading model from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        # Handle Google Drive's virus scan warning for large files
        if 'Content-Disposition' not in response.headers:
            # This might be a confirmation page, try to extract the confirm token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'confirm': value, 'id': '1CaoGi4R0u4udNHmjBh1hsJdvkdDzN0dc'}
                    response = requests.get(MODEL_URL, params=params, stream=True, timeout=60)
                    break
        
        MODEL_PATH.write_bytes(response.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def load_translator():
    global _translator
    if _translator is None:
        if not MODEL_PATH.exists():
            download_model()
        
        print(f"Loading model from {MODEL_PATH}...")
        with gzip.open(MODEL_PATH, "rb") as f:
            _translator = pickle.load(f)
        print("Translator loaded successfully.")
    return _translator


def triGramTranslate(sentence):
    """Translate sentence using trigram model"""
    translator = load_translator()
    tokens = nltk.word_tokenize(sentence.lower())
    translated = translator.tag(tokens)

    result = []
    for token, trans in translated:
        if trans == 'NNN':
            result.append(TranslaterLogic.convertText(token))
        else:
            result.append(trans)
    return " ".join(result)