import os
import json
import re
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ATSService:
    """
    ATS (Applicant Tracking System) Service for comparing resumes with job descriptions
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common ATS keywords categories
        self.skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'nodejs', 'sql', 
            'mongodb', 'postgresql', 'aws', 'azure', 'docker', 'kubernetes',
            'machine learning', 'data science', 'artificial intelligence', 'ai',
            'html', 'css', 'bootstrap', 'git', 'github', 'agile', 'scrum',
            'django', 'flask', 'spring', 'express', 'rest api', 'microservices',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'
        ]
        
        self.experience_keywords = [
            'years', 'experience', 'worked', 'developed', 'managed', 'led',
            'created', 'implemented', 'designed', 'built', 'maintained',
            'collaborated', 'coordinated', 'supervised', 'mentored'
        ]
        
        self.education_keywords = [
            'degree', 'bachelor', 'master', 'phd', 'doctorate', 'university',
            'college', 'education', 'graduated', 'gpa', 'honors', 'cum laude'
        ]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, converting to lowercase,
        tokenizing, removing stopwords, and stemming
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def extract_keywords(self, text: str, category: str = 'all') -> List[str]:
        """
        Extract relevant keywords from text based on category
        """
        text_lower = text.lower()
        found_keywords = []
        
        keyword_sets = {
            'skills': self.skill_keywords,
            'experience': self.experience_keywords,
            'education': self.education_keywords,
            'all': self.skill_keywords + self.experience_keywords + self.education_keywords
        }
        
        keywords_to_check = keyword_sets.get(category, self.skill_keywords)
        
        for keyword in keywords_to_check:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords

    def calculate_keyword_match_score(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Calculate keyword matching score between resume and job description
        """
        # Extract keywords from both texts
        resume_keywords = set(self.extract_keywords(resume_text))
        jd_keywords = set(self.extract_keywords(jd_text))
        
        # Calculate matches
        matched_keywords = resume_keywords.intersection(jd_keywords)
        missing_keywords = jd_keywords - resume_keywords
        
        # Calculate score
        if len(jd_keywords) == 0:
            keyword_score = 0
        else:
            keyword_score = len(matched_keywords) / len(jd_keywords) * 100
        
        return {
            'keyword_score': round(keyword_score, 2),
            'matched_keywords': list(matched_keywords),
            'missing_keywords': list(missing_keywords),
            'total_jd_keywords': len(jd_keywords),
            'matched_count': len(matched_keywords)
        }

    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate semantic similarity using TF-IDF and cosine similarity
        """
        # Preprocess texts
        resume_processed = self.preprocess_text(resume_text)
        jd_processed = self.preprocess_text(jd_text)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0] * 100
        
        return round(similarity_score, 2)

    def analyze_sections(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Analyze different sections of the resume against job description
        """
        sections_analysis = {}
        
        # Skills analysis
        skills_match = self.calculate_keyword_match_score(resume_text, jd_text)
        sections_analysis['skills'] = {
            'score': skills_match['keyword_score'],
            'matched': skills_match['matched_keywords'][:10],  # Top 10 matches
            'missing': skills_match['missing_keywords'][:10]   # Top 10 missing
        }
        
        # Experience keywords analysis
        resume_exp_keywords = self.extract_keywords(resume_text, 'experience')
        jd_exp_keywords = self.extract_keywords(jd_text, 'experience')
        
        exp_match_count = len(set(resume_exp_keywords).intersection(set(jd_exp_keywords)))
        exp_total = len(set(jd_exp_keywords))
        exp_score = (exp_match_count / exp_total * 100) if exp_total > 0 else 0
        
        sections_analysis['experience'] = {
            'score': round(exp_score, 2),
            'keywords_found': exp_match_count,
            'total_keywords': exp_total
        }
        
        # Education keywords analysis
        resume_edu_keywords = self.extract_keywords(resume_text, 'education')
        jd_edu_keywords = self.extract_keywords(jd_text, 'education')
        
        edu_match_count = len(set(resume_edu_keywords).intersection(set(jd_edu_keywords)))
        edu_total = len(set(jd_edu_keywords))
        edu_score = (edu_match_count / edu_total * 100) if edu_total > 0 else 50  # Default score if no edu keywords in JD
        
        sections_analysis['education'] = {
            'score': round(edu_score, 2),
            'keywords_found': edu_match_count,
            'total_keywords': edu_total
        }
        
        return sections_analysis

    def generate_ats_score(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Generate comprehensive ATS score and analysis
        """
        # Calculate different components
        keyword_analysis = self.calculate_keyword_match_score(resume_text, jd_text)
        semantic_score = self.calculate_semantic_similarity(resume_text, jd_text)
        sections_analysis = self.analyze_sections(resume_text, jd_text)
        
        # Calculate weighted overall score
        keyword_weight = 0.4
        semantic_weight = 0.3
        skills_weight = 0.2
        experience_weight = 0.1
        
        overall_score = (
            keyword_analysis['keyword_score'] * keyword_weight +
            semantic_score * semantic_weight +
            sections_analysis['skills']['score'] * skills_weight +
            sections_analysis['experience']['score'] * experience_weight
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(keyword_analysis, sections_analysis)
        
        # Determine ATS compatibility level
        if overall_score >= 80:
            compatibility = "Excellent"
            status = "High chance of passing ATS screening"
        elif overall_score >= 60:
            compatibility = "Good"
            status = "Moderate chance of passing ATS screening"
        elif overall_score >= 40:
            compatibility = "Fair"
            status = "Low chance of passing ATS screening"
        else:
            compatibility = "Poor"
            status = "Very low chance of passing ATS screening"
        
        return {
            'overall_score': round(overall_score, 2),
            'compatibility_level': compatibility,
            'status': status,
            'detailed_scores': {
                'keyword_matching': keyword_analysis['keyword_score'],
                'semantic_similarity': semantic_score,
                'skills_match': sections_analysis['skills']['score'],
                'experience_match': sections_analysis['experience']['score'],
                'education_match': sections_analysis['education']['score']
            },
            'keyword_analysis': {
                'matched_keywords': keyword_analysis['matched_keywords'],
                'missing_keywords': keyword_analysis['missing_keywords'][:15],  # Top 15 missing
                'match_percentage': keyword_analysis['keyword_score']
            },
            'sections_breakdown': sections_analysis,
            'recommendations': recommendations,
            'metadata': {
                'total_jd_keywords': keyword_analysis['total_jd_keywords'],
                'matched_keywords_count': keyword_analysis['matched_count'],
                'processing_timestamp': self.get_timestamp()
            }
        }

    def generate_recommendations(self, keyword_analysis: Dict, sections_analysis: Dict) -> List[str]:
        """
        Generate personalized recommendations based on analysis
        """
        recommendations = []
        
        # Keyword-based recommendations
        if keyword_analysis['keyword_score'] < 50:
            recommendations.append("Add more relevant keywords from the job description to your resume")
            
        if len(keyword_analysis['missing_keywords']) > 10:
            top_missing = keyword_analysis['missing_keywords'][:5]
            recommendations.append(f"Consider adding these important skills: {', '.join(top_missing)}")
        
        # Skills recommendations
        if sections_analysis['skills']['score'] < 60:
            recommendations.append("Highlight your technical skills more prominently")
            recommendations.append("Use exact skill names as mentioned in the job description")
        
        # Experience recommendations
        if sections_analysis['experience']['score'] < 50:
            recommendations.append("Use more action verbs and quantify your achievements")
            recommendations.append("Include specific examples of your work experience")
        
        # General recommendations
        recommendations.append("Use standard section headers (Experience, Education, Skills)")
        recommendations.append("Save your resume as a PDF and plain text version")
        recommendations.append("Avoid using tables, graphics, or unusual formatting")
        
        return recommendations

    def get_timestamp(self) -> str:
        """
        Get current timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """
        Extract contact information from resume
        """
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Phone pattern
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phone'] = ''.join(phones[0]) if phones else None
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        contact_info['linkedin'] = linkedin[0] if linkedin else None
        
        return contact_info