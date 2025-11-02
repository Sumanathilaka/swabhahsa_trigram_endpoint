from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import translator_fun

app = FastAPI(
    title="Sinhala Transliterator API", 
    description="Trigram-based Sinhala transliterator", 
    version="1.0"
)

class TranslationRequest(BaseModel):
    sentence: str
    
    class Config:
        schema_extra = {
            "example": {
                "sentence": "hello world"
            }
        }

@app.post("/translate")
def translate_text(request: TranslationRequest):
    """Translate a given sentence to Sinhala."""
    try:
        if not request.sentence or not request.sentence.strip():
            raise HTTPException(status_code=400, detail="Sentence cannot be empty")
        
        translation = translator_fun.triGramTranslate(request.sentence)
        return {
            "success": True,
            "original": request.sentence, 
            "translated": translation
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Translation error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/")
def home():
    return {
        "message": "Welcome to the Sinhala Transliterator API",
        "endpoints": {
            "POST /translate": "Translate English text to Sinhala",
            "GET /docs": "API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}