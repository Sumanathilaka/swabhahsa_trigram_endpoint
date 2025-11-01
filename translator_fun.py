import pickle
import nltk
import gzip
from pathlib import Path
import TranslaterLogic
import requests


APP_DIR = Path(__file__).parent.absolute()
MODEL_URL = "https://drive.google.com/file/d/1CaoGi4R0u4udNHmjBh1hsJdvkdDzN0dc/view?usp=sharing"
MODEL_PATH = APP_DIR / "trigramTrans.gz"
_translator = None

def load_translator():
    global _translator
    if _translator is None:
        if not MODEL_PATH.exists():
            print("Downloading model...")
            r = requests.get(MODEL_URL)
            MODEL_PATH.write_bytes(r.content)
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

