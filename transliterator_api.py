from fastapi import FastAPI
from pydantic import BaseModel
import Git.translator_fun as translator_fun 

app = FastAPI(title="Translator API", description="Trigram-based translator", version="1.0")

class TranslationRequest(BaseModel):
    sentence: str

@app.post("/translate")
def translate_text(request: TranslationRequest):
    """Translate a given sentence."""
    try:
        translation = translator_fun.triGramTranslate(request.sentence)
        return {"original": request.sentence, "translated": translation}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Welcome to the Translator API"}
