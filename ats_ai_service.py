import os
import json
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from typing import Dict, List, Any, Optional
from config_ats import ATSConfig

class ATSAIService:
    """
    AI-powered ATS service using Google's Gemini model for intelligent resume analysis
    """
    
    def __init__(self, project_id: str = None, location: str = None):
        print(f"[DEBUG] ATSAIService.__init__: Starting initialization")
        self.config = ATSConfig()
        self.project_id = project_id or self.config.PROJECT_ID
        self.location = location or self.config.LOCATION
        print(f"[DEBUG] ATSAIService.__init__: Using project_id={self.project_id}, location={self.location}")
        print(f"[DEBUG] ATSAIService.__init__: Credentials path={self.config.CREDENTIALS_PATH}")
        
        # Validate credentials
        if not self.config.validate_credentials():
            raise Exception(f"Credentials file not found: {self.config.CREDENTIALS_PATH}")
        
        self._init_vertex_ai()
        self.model = self._create_model()
        print(f"[DEBUG] ATSAIService.__init__: Initialization completed successfully")

    def _init_vertex_ai(self):
        """Initialize Vertex AI with project configuration"""
        print(f"[DEBUG] _init_vertex_ai: Starting Vertex AI initialization")
        try:
            if self.config.CREDENTIALS_PATH:
                print(f"[DEBUG] _init_vertex_ai: Setting GOOGLE_APPLICATION_CREDENTIALS to {self.config.CREDENTIALS_PATH}")
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.CREDENTIALS_PATH
            else:
                print(f"[DEBUG] _init_vertex_ai: No credentials path provided, using default authentication")
            
            print(f"[DEBUG] _init_vertex_ai: Calling vertexai.init with project={self.project_id}, location={self.location}")
            vertexai.init(project=self.project_id, location=self.location)
            print(f"[DEBUG] _init_vertex_ai: Vertex AI initialized successfully")
        except Exception as e:
            print(f"[ERROR] _init_vertex_ai: Failed to initialize Vertex AI: {str(e)}")
            print(f"[ERROR] _init_vertex_ai: Exception type: {type(e).__name__}")
            raise

    def _create_model(self) -> GenerativeModel:
        """Create and configure the Gemini model for ATS analysis"""
        print(f"[DEBUG] _create_model: Creating model with ID: {self.config.MODEL_ID}")
        try:
            model = GenerativeModel(
                self.config.MODEL_ID,
                system_instruction=[
                "You are an expert ATS (Applicant Tracking System) analyzer specializing in resume evaluation.",
                "Your role is to analyze resumes against job descriptions and provide detailed, objective scoring.",
                "Always respond with valid JSON structure containing numerical scores and detailed analysis.",
                "Focus on skills matching, experience relevance, education alignment, and overall fit.",
                "Provide specific recommendations for improvement based on the job requirements.",
                "Be consistent in your scoring methodology and provide clear explanations for scores.",
                "Consider industry standards and common ATS practices in your evaluation.",
                "Identify both strengths and areas for improvement in the candidate's profile."
            ]
            )
            print(f"[DEBUG] _create_model: Model created successfully")
            return model
        except Exception as e:
            print(f"[ERROR] _create_model: Failed to create model: {str(e)}")
            print(f"[ERROR] _create_model: Exception type: {type(e).__name__}")
            raise

    def _generate_ats_prompt(self, resume_text: str, job_description: str) -> str:
        """Generate a comprehensive prompt for ATS analysis"""
        return f"""
        Please analyze the following resume against the job description and provide a comprehensive ATS evaluation.

        JOB DESCRIPTION:
        {job_description}

        RESUME:
        {resume_text}

        Please provide your analysis in the following JSON format:
        {{
            "overall_score": <number between 0-100>,
            "compatibility_level": "<Excellent/Good/Fair/Poor>",
            "detailed_scores": {{
                "skills_match": <number between 0-100>,
                "experience_relevance": <number between 0-100>,
                "education_alignment": <number between 0-100>,
                "keyword_optimization": <number between 0-100>,
                "format_ats_friendly": <number between 0-100>
            }},
            "skills_analysis": {{
                "matched_skills": ["skill1", "skill2", ...],
                "missing_critical_skills": ["skill1", "skill2", ...],
                "skill_gap_score": <number between 0-100>
            }},
            "experience_analysis": {{
                "relevant_experience_years": <number>,
                "experience_match_score": <number between 0-100>,
                "key_achievements_identified": ["achievement1", "achievement2", ...],
                "experience_gaps": ["gap1", "gap2", ...]
            }},
            "education_analysis": {{
                "education_match": "<Excellent/Good/Adequate/Insufficient>",
                "relevant_certifications": ["cert1", "cert2", ...],
                "education_score": <number between 0-100>
            }},
            "keyword_analysis": {{
                "keyword_density_score": <number between 0-100>,
                "important_keywords_found": ["keyword1", "keyword2", ...],
                "missing_keywords": ["keyword1", "keyword2", ...],
                "keyword_suggestions": ["suggestion1", "suggestion2", ...]
            }},
            "recommendations": [
                "recommendation1",
                "recommendation2",
                "recommendation3",
                ...
            ],
            "strengths": [
                "strength1",
                "strength2",
                "strength3"
            ],
            "improvement_areas": [
                "area1",
                "area2",
                "area3"
            ],
            "ats_compatibility": {{
                "format_score": <number between 0-100>,
                "parsing_friendliness": "<Excellent/Good/Fair/Poor>",
                "format_recommendations": ["rec1", "rec2", ...]
            }},
            "industry_fit": {{
                "industry_alignment_score": <number between 0-100>,
                "relevant_industry_experience": <boolean>,
                "transferable_skills": ["skill1", "skill2", ...]
            }}
        }}

        Ensure all scores are realistic and well-justified. Provide specific, actionable recommendations.
        """

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Analyze a resume against a job description using Gemini AI
        
        Args:
            resume_text: The text content of the resume
            job_description: The job description to match against
            
        Returns:
            Comprehensive ATS analysis results
        """
        print(f"[DEBUG] analyze_resume: Starting analysis")
        print(f"[DEBUG] analyze_resume: Resume text length: {len(resume_text)} characters")
        print(f"[DEBUG] analyze_resume: Job description length: {len(job_description)} characters")
        
        try:
            # Generate the analysis prompt
            print(f"[DEBUG] analyze_resume: Generating prompt")
            prompt = self._generate_ats_prompt(resume_text, job_description)
            print(f"[DEBUG] analyze_resume: Prompt generated, length: {len(prompt)} characters")
            
            # Configure generation parameters
            print(f"[DEBUG] analyze_resume: Configuring generation parameters")
            generation_config = GenerationConfig(
                temperature=self.config.GEMINI_CONFIG['temperature'],
                top_p=self.config.GEMINI_CONFIG['top_p'],
                candidate_count=self.config.GEMINI_CONFIG['candidate_count'],
                max_output_tokens=self.config.GEMINI_CONFIG['max_output_tokens'],
            )
            print(f"[DEBUG] analyze_resume: Generation config created")
            
            # Generate response from Gemini
            print(f"[DEBUG] analyze_resume: Calling Gemini model.generate_content")
            response = self.model.generate_content(
                [prompt], 
                generation_config=generation_config
            )
            print(f"[DEBUG] analyze_resume: Gemini response received")
            print(f"[DEBUG] analyze_resume: Response text length: {len(response.text) if hasattr(response, 'text') and response.text else 0} characters")
            
            # Parse the JSON response
            try:
                print(f"[DEBUG] analyze_resume: Parsing JSON response")
                if not response.text:
                    print(f"[ERROR] analyze_resume: Empty response text from Gemini")
                    return self._create_fallback_response("Empty response from AI model")
                
                print(f"[DEBUG] analyze_resume: Response text preview: {response.text[:200]}...")
                response = response.text.replace('```json', '')
                response = response.replace('```', '')
                response = response.replace('\\', '')
                analysis_result = json.loads(response)
                print(f"[DEBUG] analyze_resume: JSON parsing successful")
                
                # Add metadata
                print(f"[DEBUG] analyze_resume: Adding metadata")
                analysis_result['metadata'] = {
                    'model_used': self.config.MODEL_ID,
                    'analysis_timestamp': self._get_timestamp(),
                    'processing_method': 'gemini_ai'
                }
                
                # Validate and normalize scores
                print(f"[DEBUG] analyze_resume: Validating and normalizing scores")
                analysis_result = self._validate_and_normalize_scores(analysis_result)
                
                # Transform to match UI expectations
                print(f"[DEBUG] analyze_resume: Transforming response for UI compatibility")
                analysis_result = self._transform_for_ui(analysis_result)
                print(f"[DEBUG] analyze_resume: Analysis completed successfully")
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create a fallback response
                print(f"[ERROR] analyze_resume: JSON parsing failed: {str(e)}")
                print(f"[ERROR] analyze_resume: Raw response text: {response.text}")
                return self._create_fallback_response(f"JSON parsing error: {str(e)}")
                
        except Exception as e:
            print(f"[ERROR] analyze_resume: Analysis failed with exception: {str(e)}")
            print(f"[ERROR] analyze_resume: Exception type: {type(e).__name__}")
            import traceback
            print(f"[ERROR] analyze_resume: Full traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Analysis error: {str(e)}")

    def _validate_and_normalize_scores(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize all scores to be within 0-100 range"""
        
        def normalize_score(score):
            """Ensure score is between 0 and 100"""
            if isinstance(score, (int, float)):
                return max(0, min(100, score))
            return 0
        
        # Normalize main scores
        if 'overall_score' in analysis:
            analysis['overall_score'] = normalize_score(analysis['overall_score'])
        
        if 'detailed_scores' in analysis:
            for key, value in analysis['detailed_scores'].items():
                analysis['detailed_scores'][key] = normalize_score(value)
        
        # Set compatibility level based on overall score
        overall_score = analysis.get('overall_score', 0)
        if overall_score >= self.config.EXCELLENT_SCORE_THRESHOLD:
            analysis['compatibility_level'] = "Excellent"
            analysis['status'] = "High chance of passing ATS screening"
        elif overall_score >= self.config.GOOD_SCORE_THRESHOLD:
            analysis['compatibility_level'] = "Good"
            analysis['status'] = "Moderate chance of passing ATS screening"
        elif overall_score >= self.config.FAIR_SCORE_THRESHOLD:
            analysis['compatibility_level'] = "Fair"
            analysis['status'] = "Low chance of passing ATS screening"
        else:
            analysis['compatibility_level'] = "Poor"
            analysis['status'] = "Very low chance of passing ATS screening"
        
        return analysis

    def _transform_for_ui(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Transform AI response to match UI expectations"""
        try:
            # Transform detailed_scores to match UI expectations
            if 'detailed_scores' in analysis:
                detailed_scores = analysis['detailed_scores']
                # Map AI response keys to UI expected keys
                transformed_scores = {
                    'keyword_matching': detailed_scores.get('keyword_optimization', 0),
                    'semantic_similarity': detailed_scores.get('skills_match', 0),
                    'skills_match': detailed_scores.get('skills_match', 0),
                    'experience_match': detailed_scores.get('experience_relevance', 0),
                    'education_match': detailed_scores.get('education_alignment', 0)
                }
                analysis['detailed_scores'] = transformed_scores
            
            # Transform keyword_analysis to match UI expectations
            if 'keyword_analysis' in analysis:
                keyword_analysis = analysis['keyword_analysis']
                transformed_keywords = {
                    'matched_keywords': keyword_analysis.get('important_keywords_found', []),
                    'missing_keywords': keyword_analysis.get('missing_keywords', []),
                    'match_percentage': keyword_analysis.get('keyword_density_score', 0)
                }
                analysis['keyword_analysis'] = transformed_keywords
            
            # Ensure recommendations exist
            if 'recommendations' not in analysis:
                analysis['recommendations'] = []
            
            # Add status if not present
            if 'status' not in analysis:
                overall_score = analysis.get('overall_score', 0)
                if overall_score >= 80:
                    analysis['status'] = "High chance of passing ATS screening"
                elif overall_score >= 60:
                    analysis['status'] = "Moderate chance of passing ATS screening"
                elif overall_score >= 40:
                    analysis['status'] = "Low chance of passing ATS screening"
                else:
                    analysis['status'] = "Very low chance of passing ATS screening"
            
            print(f"[DEBUG] _transform_for_ui: Transformation completed successfully")
            return analysis
            
        except Exception as e:
            print(f"[ERROR] _transform_for_ui: Transformation failed: {str(e)}")
            # Return original analysis if transformation fails
            return analysis

    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback response when AI analysis fails"""
        return {
            'overall_score': 0,
            'compatibility_level': "Poor",
            'status': "Analysis failed - manual review required",
            'error': error_message,
            'detailed_scores': {
                'skills_match': 0,
                'experience_relevance': 0,
                'education_alignment': 0,
                'keyword_optimization': 0,
                'format_ats_friendly': 0
            },
            'recommendations': [
                "AI analysis failed. Please try again or use manual review.",
                "Ensure resume and job description are properly formatted.",
                "Check network connectivity and API credentials."
            ],
            'metadata': {
                'analysis_timestamp': self._get_timestamp(),
                'processing_method': 'fallback',
                'error': error_message
            }
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            'error': True,
            'message': error_message,
            'overall_score': 0,
            'compatibility_level': "Error",
            'status': "Analysis failed due to technical error",
            'metadata': {
                'analysis_timestamp': self._get_timestamp(),
                'processing_method': 'error_handler'
            }
        }

    def batch_analyze_resumes(self, resumes_data: List[Dict[str, str]], job_description: str) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes against a single job description
        
        Args:
            resumes_data: List of dictionaries containing resume text and metadata
            job_description: The job description to match against
            
        Returns:
            List of analysis results for each resume
        """
        results = []
        
        for i, resume_data in enumerate(resumes_data):
            try:
                resume_text = resume_data.get('text', '')
                resume_id = resume_data.get('id', f'resume_{i+1}')
                
                analysis = self.analyze_resume(resume_text, job_description)
                analysis['resume_id'] = resume_id
                analysis['batch_position'] = i + 1
                
                results.append(analysis)
                
            except Exception as e:
                error_result = self._create_error_response(f"Batch analysis error for resume {i+1}: {str(e)}")
                error_result['resume_id'] = resume_data.get('id', f'resume_{i+1}')
                error_result['batch_position'] = i + 1
                results.append(error_result)
        
        return results

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model configuration"""
        return {
            'model_id': self.config.MODEL_ID,
            'project_id': self.project_id,
            'location': self.location,
            'service_type': 'gemini_vertex_ai'
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()