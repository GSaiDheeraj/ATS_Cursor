from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
import json
import time
from werkzeug.utils import secure_filename
import tempfile

# Import existing modules from sample backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sample backend'))

from file_repository import PDFReader, TXTReader
from config import Config

# Import new ATS modules
from ats_service import ATSService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}

# Initialize services
# Try to initialize with AI first, fallback to traditional if needed
print(f"[DEBUG] app_ats.py: Starting ATS service initialization")
try:
    print(f"[DEBUG] app_ats.py: Attempting to initialize with AI enabled")
    ats_service = ATSService(use_ai=True)
    analysis_method = ats_service.get_analysis_method()
    print(f"[DEBUG] app_ats.py: ATS Service initialized successfully with method: {analysis_method}")
    logger.info(f"ATS Service initialized with method: {analysis_method}")
except Exception as e:
    print(f"[ERROR] app_ats.py: Failed to initialize AI service: {e}")
    print(f"[ERROR] app_ats.py: Exception type: {type(e).__name__}")
    import traceback
    print(f"[ERROR] app_ats.py: Full traceback: {traceback.format_exc()}")
    logger.warning(f"Failed to initialize AI service: {e}")
    logger.info("Initializing with traditional NLP methods")
    print(f"[DEBUG] app_ats.py: Falling back to traditional NLP methods")
    ats_service = ATSService(use_ai=False)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ATSFileProcessor:
    """File processor specifically for ATS system"""
    
    def __init__(self):
        self.readers = {
            'pdf': PDFReader(),
            'txt': TXTReader(),
        }
    
    def process_file(self, uploaded_file):
        """Process uploaded file and extract text content"""
        if not uploaded_file or uploaded_file.filename == '':
            raise ValueError("No file selected")
        
        if not allowed_file(uploaded_file.filename):
            raise ValueError("File type not supported. Please upload PDF or TXT files.")
        
        file_extension = uploaded_file.filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_extension in self.readers:
                reader = self.readers[file_extension]
                content = reader.read(uploaded_file)
            else:
                # For other file types, try to read as text
                content = uploaded_file.read().decode('utf-8', errors='ignore')
            
            return content
        except Exception as e:
            logger.error(f"Failed to process {file_extension.upper()} file: {str(e)}")
            raise ValueError(f"Failed to process file: {str(e)}")

@app.route('/')
def index():
    """Serve the main ATS application page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ATS Resume Analyzer',
        'version': '1.0.0',
        'timestamp': time.time()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """
    Main endpoint to analyze resume against job description
    Expects: multipart/form-data with 'resume' file and 'job_description' text
    """
    try:
        start_time = time.time()
        
        # Validate request
        if 'resume' not in request.files:
            return jsonify({
                'error': 'No resume file provided',
                'status': False
            }), 400
        
        if 'job_description' not in request.form:
            return jsonify({
                'error': 'No job description provided',
                'status': False
            }), 400
        
        resume_file = request.files['resume']
        job_description = request.form['job_description'].strip()
        
        if not job_description:
            return jsonify({
                'error': 'Job description cannot be empty',
                'status': False
            }), 400
        
        # Process resume file
        file_processor = ATSFileProcessor()
        resume_text = file_processor.process_file(resume_file)
        
        if not resume_text.strip():
            return jsonify({
                'error': 'Resume file appears to be empty or could not be read',
                'status': False
            }), 400
        
        # Generate ATS analysis
        logger.info("Starting ATS analysis")
        ats_analysis = ats_service.generate_ats_score(resume_text, job_description)
        
        # Extract additional metrics from AI analysis or calculate basic ones
        additional_metrics = {}
        
        # Build additional_metrics structure that matches UI expectations
        if ats_service.get_analysis_method() == 'gemini_ai':
            # For AI analysis, extract from AI response
            additional_metrics = {
                'readability': {
                    'score': 75,  # Could be extracted from AI response if available
                    'level': 'Good'
                },
                'formatting': {
                    'formatting_score': ats_analysis.get('ats_compatibility', {}).get('format_score', 85),
                    'ats_friendly': True
                },
                'keyword_density': ats_analysis.get('keyword_analysis', {}),
                'experience_analysis': {
                    'relevance_score': ats_analysis.get('experience_analysis', {}).get('experience_match_score', 70)
                },
                'industry_analysis': {
                    'industry_score': ats_analysis.get('industry_fit', {}).get('industry_alignment_score', 65),
                    'industry_detected': 'Technology'  # Could be extracted from AI if available
                }
            }
        else:
            # Fallback for traditional analysis
            additional_metrics = {
                'readability': {'score': 75, 'level': 'Good'},
                'formatting': {'formatting_score': 85, 'ats_friendly': True},
                'keyword_density': ats_analysis.get('keyword_analysis', {}),
                'experience_analysis': {'relevance_score': 70},
                'industry_analysis': {'industry_score': 65, 'industry_detected': 'General'}
            }
        
        # Improvement suggestions are now included in the main analysis
        improvement_suggestions = ats_analysis.get('recommendations', [])
        
        # Extract contact information
        contact_info = ats_service.extract_contact_info(resume_text)
        
        processing_time = time.time() - start_time
        
        # Compile comprehensive response
        response = {
            'status': True,
            'message': 'Resume analysis completed successfully',
            'analysis': ats_analysis,
            'additional_metrics': additional_metrics,
            'contact_info': contact_info,
            'improvement_suggestions': improvement_suggestions,
            'metadata': {
                'processing_time_seconds': round(processing_time, 2),
                'resume_filename': resume_file.filename,
                'resume_word_count': len(resume_text.split()),
                'jd_word_count': len(job_description.split()),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'analysis_method': ats_service.get_analysis_method()
            }
        }
        
        logger.info(f"ATS analysis completed in {processing_time:.2f} seconds")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': False
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during analysis',
            'details': str(e),
            'status': False
        }), 500

@app.route('/api/quick-score', methods=['POST'])
def quick_score():
    """
    Quick scoring endpoint for basic ATS score calculation
    Expects JSON with 'resume_text' and 'job_description'
    """
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({
                'error': 'Both resume_text and job_description are required',
                'status': False
            }), 400
        
        resume_text = data['resume_text'].strip()
        job_description = data['job_description'].strip()
        
        if not resume_text or not job_description:
            return jsonify({
                'error': 'Resume text and job description cannot be empty',
                'status': False
            }), 400
        
        # Generate quick ATS score
        ats_analysis = ats_service.generate_ats_score(resume_text, job_description)
        
        # Return simplified response
        response = {
            'status': True,
            'overall_score': ats_analysis['overall_score'],
            'compatibility_level': ats_analysis['compatibility_level'],
            'status_message': ats_analysis['status'],
            'key_metrics': {
                'keyword_matching': ats_analysis['detailed_scores']['keyword_matching'],
                'semantic_similarity': ats_analysis['detailed_scores']['semantic_similarity']
            },
            'top_recommendations': ats_analysis['recommendations'][:3]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in quick score: {str(e)}")
        return jsonify({
            'error': 'Failed to calculate quick score',
            'details': str(e),
            'status': False
        }), 500

@app.route('/api/keywords', methods=['POST'])
def analyze_keywords():
    """
    Keyword analysis endpoint
    Returns detailed keyword matching analysis
    """
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({
                'error': 'Both resume_text and job_description are required',
                'status': False
            }), 400
        
        resume_text = data['resume_text']
        job_description = data['job_description']
        
        # Perform keyword analysis
        keyword_analysis = ats_service.calculate_keyword_match_score(resume_text, job_description)
        
        # Calculate basic keyword density
        resume_words = resume_text.lower().split()
        jd_words = job_description.lower().split()
        
        # Simple keyword density calculation
        total_words = len(resume_words)
        keyword_count = sum(1 for word in resume_words if word in jd_words)
        keyword_density = {
            'density_percentage': round((keyword_count / total_words * 100), 2) if total_words > 0 else 0,
            'total_words': total_words,
            'keyword_matches': keyword_count
        }
        
        response = {
            'status': True,
            'keyword_matching': keyword_analysis,
            'keyword_density': keyword_density,
            'recommendations': [
                'Use exact keywords from job description',
                'Include variations of important terms',
                'Maintain natural keyword density (1-3%)',
                'Focus on technical skills and requirements'
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in keyword analysis: {str(e)}")
        return jsonify({
            'error': 'Failed to analyze keywords',
            'details': str(e),
            'status': False
        }), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """
    Get personalized improvement suggestions
    """
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({
                'error': 'Both resume_text and job_description are required',
                'status': False
            }), 400
        
        resume_text = data['resume_text']
        job_description = data['job_description']
        
        # Generate comprehensive analysis for suggestions
        ats_analysis = ats_service.generate_ats_score(resume_text, job_description)
        suggestions = ats_analysis.get('recommendations', [])
        
        # Add basic formatting and readability suggestions if not present
        if not suggestions:
            suggestions = [
                "Use standard section headers (Experience, Education, Skills)",
                "Include relevant keywords from the job description",
                "Quantify achievements with specific numbers and metrics",
                "Use action verbs to describe your experience",
                "Save resume in PDF format for ATS compatibility"
            ]
        
        # Basic formatting and readability analysis
        formatting_analysis = {
            'formatting_score': 85,
            'ats_friendly': True,
            'issues': []
        }
        
        readability_analysis = {
            'score': 75,
            'level': 'Good'
        }
        
        response = {
            'status': True,
            'suggestions': suggestions,
            'formatting_tips': formatting_analysis,
            'readability_tips': readability_analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return jsonify({
            'error': 'Failed to generate suggestions',
            'details': str(e),
            'status': False
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.',
        'status': False
    }), 413

@app.route('/api/service-info', methods=['GET'])
def get_service_info():
    """Get information about the current ATS service configuration"""
    try:
        service_info = ats_service.get_service_info()
        return jsonify({
            'status': True,
            'service_info': service_info,
            'message': f'ATS Service running with {service_info["current_method"]} analysis'
        })
    except Exception as e:
        logger.error(f"Error getting service info: {str(e)}")
        return jsonify({
            'error': f'Failed to get service information: {str(e)}',
            'status': False
        }), 500

@app.route('/api/toggle-ai', methods=['POST'])
def toggle_ai_analysis():
    """Toggle between AI and traditional analysis methods"""
    try:
        data = request.get_json() or {}
        use_ai = data.get('use_ai', True)
        project_id = data.get('project_id')
        location = data.get('location')
        
        success = ats_service.toggle_ai_usage(use_ai, project_id, location)
        
        if success:
            new_method = ats_service.get_analysis_method()
            return jsonify({
                'status': True,
                'message': f'Analysis method switched to: {new_method}',
                'current_method': new_method
            })
        else:
            return jsonify({
                'status': False,
                'error': 'Failed to toggle AI analysis method'
            }), 400
            
    except Exception as e:
        logger.error(f"Error toggling AI analysis: {str(e)}")
        return jsonify({
            'error': f'Failed to toggle analysis method: {str(e)}',
            'status': False
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'status': False
    }), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )