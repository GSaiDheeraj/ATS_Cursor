#!/usr/bin/env python3
"""
Basic test script for ATS Resume Analyzer
Tests core functionality without requiring file uploads
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from ats_service import ATSService
        from ats_scoring import ATSScoring
        print("✅ ATS modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ats_service():
    """Test ATS service functionality"""
    print("\n🧪 Testing ATS Service...")
    
    try:
        # Initialize service
        ats_service = ATSService()
        
        # Sample resume text
        resume_text = """
        John Doe
        Software Engineer
        
        Experience:
        - 3 years of experience in Python development
        - Worked with Flask, Django, and FastAPI
        - Experience with machine learning and data science
        - Built REST APIs and web applications
        
        Skills:
        Python, JavaScript, SQL, Git, Docker, AWS, React
        
        Education:
        Bachelor's degree in Computer Science
        """
        
        # Sample job description
        jd_text = """
        We are looking for a Software Engineer with:
        - 2+ years of Python experience
        - Experience with Flask or Django
        - Knowledge of machine learning
        - SQL and database experience
        - Cloud experience (AWS preferred)
        - Bachelor's degree in Computer Science or related field
        """
        
        # Test keyword extraction
        keywords = ats_service.extract_keywords(resume_text)
        print(f"✅ Extracted {len(keywords)} keywords from resume")
        
        # Test keyword matching
        keyword_match = ats_service.calculate_keyword_match_score(resume_text, jd_text)
        print(f"✅ Keyword match score: {keyword_match['keyword_score']}%")
        
        # Test semantic similarity
        similarity = ats_service.calculate_semantic_similarity(resume_text, jd_text)
        print(f"✅ Semantic similarity: {similarity}%")
        
        # Test full ATS analysis
        analysis = ats_service.generate_ats_score(resume_text, jd_text)
        print(f"✅ Overall ATS score: {analysis['overall_score']}%")
        print(f"✅ Compatibility level: {analysis['compatibility_level']}")
        
        return True
    except Exception as e:
        print(f"❌ ATS Service test error: {e}")
        return False

def test_ats_scoring():
    """Test ATS scoring functionality"""
    print("\n🧪 Testing ATS Scoring...")
    
    try:
        # Initialize scoring
        ats_scoring = ATSScoring()
        
        # Sample text
        sample_text = """
        This is a sample resume text for testing readability and formatting.
        It contains multiple sentences to test various scoring algorithms.
        The text should be analyzed for ATS compatibility and optimization.
        """
        
        # Test industry detection
        jd_text = "We need a software engineer with Python and machine learning experience"
        industry = ats_scoring.detect_industry(jd_text)
        print(f"✅ Detected industry: {industry}")
        
        # Test readability score
        readability = ats_scoring.calculate_readability_score(sample_text)
        print(f"✅ Readability score: {readability['flesch_score']}")
        
        # Test formatting score
        formatting = ats_scoring.calculate_formatting_score(sample_text)
        print(f"✅ Formatting score: {formatting['formatting_score']}%")
        
        return True
    except Exception as e:
        print(f"❌ ATS Scoring test error: {e}")
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("\n🧪 Testing Flask App...")
    
    try:
        # Import Flask modules
        from flask import Flask
        
        # Test app creation (without running)
        app = Flask(__name__)
        print("✅ Flask app can be created")
        
        # Test if our app file can be imported
        import app_ats
        print("✅ ATS Flask app module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Flask app test error: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🧪 Testing Dependencies...")
    
    required_modules = [
        'flask',
        'sklearn',
        'nltk',
        'numpy',
        'pandas',
        'werkzeug'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements_ats.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 ATS Resume Analyzer - System Test")
    print("="*50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("ATS Service", test_ats_service),
        ("ATS Scoring", test_ats_scoring),
        ("Flask App", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ATS system is ready to use.")
        print("\nTo start the application, run:")
        print("python app_ats.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_ats.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)