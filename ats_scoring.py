"""
Advanced ATS Scoring Module
Provides detailed scoring algorithms and metrics for ATS analysis
"""

import re
import math
from typing import Dict, List, Tuple, Any
from collections import Counter
import statistics

class ATSScoring:
    """
    Advanced scoring algorithms for ATS analysis
    """
    
    def __init__(self):
        self.industry_weights = {
            'technology': {
                'technical_skills': 0.4,
                'experience': 0.3,
                'education': 0.2,
                'certifications': 0.1
            },
            'healthcare': {
                'certifications': 0.35,
                'experience': 0.35,
                'education': 0.2,
                'technical_skills': 0.1
            },
            'finance': {
                'certifications': 0.3,
                'experience': 0.3,
                'education': 0.25,
                'technical_skills': 0.15
            },
            'general': {
                'experience': 0.35,
                'technical_skills': 0.25,
                'education': 0.25,
                'certifications': 0.15
            }
        }
        
        # Common industry keywords
        self.industry_keywords = {
            'technology': ['software', 'development', 'programming', 'coding', 'algorithm', 
                          'database', 'api', 'framework', 'cloud', 'devops', 'agile'],
            'healthcare': ['patient', 'medical', 'clinical', 'healthcare', 'treatment', 
                          'diagnosis', 'therapy', 'nursing', 'pharmaceutical'],
            'finance': ['financial', 'investment', 'banking', 'accounting', 'audit', 
                       'risk', 'compliance', 'trading', 'portfolio', 'analysis'],
            'marketing': ['marketing', 'campaign', 'brand', 'social media', 'seo', 
                         'analytics', 'content', 'advertising', 'promotion'],
            'sales': ['sales', 'revenue', 'client', 'customer', 'negotiation', 
                     'relationship', 'target', 'quota', 'pipeline']
        }

    def detect_industry(self, jd_text: str) -> str:
        """
        Detect the industry based on job description keywords
        """
        jd_lower = jd_text.lower()
        industry_scores = {}
        
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in jd_lower)
            industry_scores[industry] = score
        
        if max(industry_scores.values()) == 0:
            return 'general'
        
        return max(industry_scores, key=industry_scores.get)

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for ATS optimization
        """
        # Basic text metrics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if not sentences or not words:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'readability_level': 'Poor'}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        
        # Readability level
        if flesch_score >= 90:
            level = 'Excellent'
        elif flesch_score >= 80:
            level = 'Good'
        elif flesch_score >= 70:
            level = 'Fair'
        elif flesch_score >= 60:
            level = 'Acceptable'
        else:
            level = 'Poor'
        
        return {
            'flesch_score': round(flesch_score, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability_level': level
        }

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified algorithm)
        """
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for i, char in enumerate(word):
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        return max(1, syllable_count)

    def calculate_formatting_score(self, text: str) -> Dict[str, Any]:
        """
        Analyze resume formatting for ATS compatibility
        """
        formatting_issues = []
        score = 100
        
        # Check for common formatting issues
        if re.search(r'[^\x00-\x7F]', text):
            formatting_issues.append("Contains non-ASCII characters")
            score -= 10
        
        if len(re.findall(r'\t', text)) > 5:
            formatting_issues.append("Excessive use of tabs")
            score -= 5
        
        if len(re.findall(r'  +', text)) > 20:
            formatting_issues.append("Excessive whitespace")
            score -= 5
        
        # Check for standard sections
        standard_sections = ['experience', 'education', 'skills', 'summary']
        found_sections = []
        
        for section in standard_sections:
            if re.search(rf'\b{section}\b', text, re.IGNORECASE):
                found_sections.append(section)
        
        section_score = (len(found_sections) / len(standard_sections)) * 30
        score = score - 30 + section_score
        
        return {
            'formatting_score': max(0, round(score, 2)),
            'issues': formatting_issues,
            'sections_found': found_sections,
            'ats_friendly': score >= 80
        }

    def calculate_keyword_density(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Calculate keyword density and optimization metrics
        """
        # Extract important keywords from JD
        jd_words = re.findall(r'\b\w+\b', jd_text.lower())
        jd_word_freq = Counter(jd_words)
        
        # Filter out common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                       'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were',
                       'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could',
                       'should', 'may', 'might', 'must', 'can', 'do', 'does', 'did'}
        
        # Get top keywords from JD (excluding common words)
        important_jd_keywords = {word: freq for word, freq in jd_word_freq.most_common(50) 
                               if word not in common_words and len(word) > 2}
        
        # Calculate keyword density in resume
        resume_words = re.findall(r'\b\w+\b', resume_text.lower())
        resume_word_freq = Counter(resume_words)
        
        keyword_analysis = {}
        total_resume_words = len(resume_words)
        
        for keyword, jd_freq in important_jd_keywords.items():
            resume_freq = resume_word_freq.get(keyword, 0)
            density = (resume_freq / total_resume_words) * 100 if total_resume_words > 0 else 0
            
            keyword_analysis[keyword] = {
                'jd_frequency': jd_freq,
                'resume_frequency': resume_freq,
                'density_percentage': round(density, 3),
                'importance_score': jd_freq * (1 if resume_freq > 0 else 0)
            }
        
        # Calculate overall keyword optimization score
        matched_keywords = sum(1 for data in keyword_analysis.values() if data['resume_frequency'] > 0)
        optimization_score = (matched_keywords / len(important_jd_keywords)) * 100 if important_jd_keywords else 0
        
        return {
            'optimization_score': round(optimization_score, 2),
            'total_important_keywords': len(important_jd_keywords),
            'matched_keywords': matched_keywords,
            'keyword_details': dict(list(keyword_analysis.items())[:20]),  # Top 20 for display
            'average_density': round(statistics.mean([data['density_percentage'] 
                                                   for data in keyword_analysis.values()]), 3)
        }

    def calculate_experience_relevance(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Analyze experience relevance and years calculation
        """
        # Extract years of experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?'
        ]
        
        resume_years = []
        jd_years = []
        
        # Extract from resume
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    resume_years.extend([int(x) for x in match if x.isdigit()])
                else:
                    resume_years.append(int(match))
        
        # Extract from job description
        for pattern in experience_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    jd_years.extend([int(x) for x in match if x.isdigit()])
                else:
                    jd_years.append(int(match))
        
        # Calculate experience match score
        if not jd_years:
            experience_score = 75  # Default score if no specific years mentioned
        elif not resume_years:
            experience_score = 30  # Low score if resume doesn't mention years
        else:
            max_resume_years = max(resume_years)
            min_jd_years = min(jd_years)
            
            if max_resume_years >= min_jd_years:
                experience_score = 100
            else:
                # Partial score based on how close they are
                experience_score = min(100, (max_resume_years / min_jd_years) * 100)
        
        return {
            'experience_score': round(experience_score, 2),
            'resume_years_mentioned': resume_years,
            'jd_years_required': jd_years,
            'meets_requirement': experience_score >= 80
        }

    def generate_improvement_suggestions(self, analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate specific improvement suggestions based on analysis
        """
        suggestions = []
        
        # Overall score suggestions
        overall_score = analysis_results.get('overall_score', 0)
        if overall_score < 60:
            suggestions.append({
                'category': 'Critical',
                'suggestion': 'Your resume needs significant improvements to pass ATS screening',
                'priority': 'High'
            })
        
        # Keyword suggestions
        keyword_score = analysis_results.get('detailed_scores', {}).get('keyword_matching', 0)
        if keyword_score < 50:
            suggestions.append({
                'category': 'Keywords',
                'suggestion': 'Add more relevant keywords from the job description throughout your resume',
                'priority': 'High'
            })
        
        # Missing keywords
        missing_keywords = analysis_results.get('keyword_analysis', {}).get('missing_keywords', [])
        if len(missing_keywords) > 10:
            top_missing = missing_keywords[:5]
            suggestions.append({
                'category': 'Keywords',
                'suggestion': f'Consider adding these key skills: {", ".join(top_missing)}',
                'priority': 'Medium'
            })
        
        # Formatting suggestions
        suggestions.append({
            'category': 'Formatting',
            'suggestion': 'Use standard section headers: Summary, Experience, Education, Skills',
            'priority': 'Medium'
        })
        
        suggestions.append({
            'category': 'Formatting',
            'suggestion': 'Save resume in both PDF and plain text formats for different ATS systems',
            'priority': 'Low'
        })
        
        # Experience suggestions
        experience_score = analysis_results.get('detailed_scores', {}).get('experience_match', 0)
        if experience_score < 60:
            suggestions.append({
                'category': 'Experience',
                'suggestion': 'Use more action verbs and quantify your achievements with numbers',
                'priority': 'High'
            })
        
        return suggestions

    def calculate_industry_specific_score(self, resume_text: str, jd_text: str, industry: str = None) -> Dict[str, Any]:
        """
        Calculate industry-specific ATS score
        """
        if not industry:
            industry = self.detect_industry(jd_text)
        
        weights = self.industry_weights.get(industry, self.industry_weights['general'])
        
        # Calculate component scores (simplified for this example)
        technical_score = 75  # This would be calculated based on technical keywords
        experience_score = 80   # This would be calculated based on experience analysis
        education_score = 70    # This would be calculated based on education keywords
        certification_score = 60  # This would be calculated based on certifications
        
        # Calculate weighted score
        industry_score = (
            technical_score * weights['technical_skills'] +
            experience_score * weights['experience'] +
            education_score * weights['education'] +
            certification_score * weights['certifications']
        )
        
        return {
            'industry_detected': industry,
            'industry_score': round(industry_score, 2),
            'component_scores': {
                'technical_skills': technical_score,
                'experience': experience_score,
                'education': education_score,
                'certifications': certification_score
            },
            'weights_used': weights
        }