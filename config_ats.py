import os
from dotenv import load_dotenv

load_dotenv()

class ATSConfig:
    """Configuration class for ATS system with Gemini AI integration"""
    
    # Gemini/Vertex AI Configuration
    PROJECT_ID = os.getenv('PROJECT_ID')
    LOCATION = os.getenv('LOCATION', 'us-central1')
    CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    MODEL_ID = os.getenv('MODEL_ID')
    
    # Check if credentials file exists
    @classmethod
    def validate_credentials(cls):
        """Validate that the credentials file exists"""
        if cls.CREDENTIALS_PATH and not os.path.exists(cls.CREDENTIALS_PATH):
            print(f"[WARNING] Credentials file not found: {cls.CREDENTIALS_PATH}")
            return False
        return True
    
    # ATS Specific Configuration
    ATS_SCORING_WEIGHTS = {
        'skills_match': 0.35,
        'experience_match': 0.25,
        'education_match': 0.15,
        'semantic_similarity': 0.15,
        'keyword_density': 0.10
    }
    
    # Gemini Generation Configuration
    GEMINI_CONFIG = {
        'temperature': 0.1,  # Low temperature for consistent analysis
        'top_p': 0.8,
        'candidate_count': 1,
        'max_output_tokens': 4096,
    }
    
    # ATS Analysis Configuration
    MIN_SCORE_THRESHOLD = 60
    EXCELLENT_SCORE_THRESHOLD = 80
    GOOD_SCORE_THRESHOLD = 60
    FAIR_SCORE_THRESHOLD = 40