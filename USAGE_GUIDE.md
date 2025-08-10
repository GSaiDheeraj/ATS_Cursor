# ATS Resume Analyzer - Usage Guide

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
# Run the automated setup
python setup_ats.py

# Or manual setup:
pip install -r requirements_ats.txt
```

### 2. Start the Application
```bash
python app_ats.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5000`

## 📋 How to Use

### Step 1: Upload Resume
- Click the upload area or drag & drop your resume
- Supported formats: PDF, TXT, DOC, DOCX
- Maximum file size: 16MB

### Step 2: Paste Job Description
- Copy the complete job description from the job posting
- Paste it in the text area
- Include requirements, skills, and qualifications

### Step 3: Analyze
- Click "Analyze Resume" button
- Wait for processing (typically 2-5 seconds)
- Review your comprehensive ATS report

## 📊 Understanding Your Results

### Overall Score
- **80-100%**: Excellent - High chance of passing ATS
- **60-79%**: Good - Moderate chance of passing ATS  
- **40-59%**: Fair - Low chance of passing ATS
- **0-39%**: Poor - Very low chance of passing ATS

### Score Breakdown
1. **Keyword Matching** (40% weight): How well your resume matches job keywords
2. **Semantic Similarity** (30% weight): Content similarity using AI
3. **Skills Match** (20% weight): Technical and soft skills alignment
4. **Experience Match** (10% weight): Experience level relevance

### Keywords Analysis
- **Matched Keywords**: Keywords found in both resume and job description
- **Missing Keywords**: Important keywords from job description not in resume
- **Recommendations**: Specific keywords to add to your resume

## 🎯 Optimization Tips

### High-Impact Changes
1. **Add Missing Keywords**: Include relevant keywords from the job description
2. **Use Exact Terms**: Match the exact phrasing used in the job posting
3. **Quantify Achievements**: Use numbers and metrics (e.g., "Increased sales by 25%")
4. **Standard Headers**: Use conventional section names (Experience, Education, Skills)

### Formatting Best Practices
- Use simple, clean formatting
- Avoid tables, graphics, or images
- Use standard fonts (Arial, Calibri, Times New Roman)
- Save as PDF for consistency
- Keep file size under 16MB

### Content Optimization
- Include relevant skills and technologies
- Use action verbs (managed, developed, implemented)
- Match experience level requirements
- Include industry-specific terminology
- Tailor content for each application

## 🔧 Advanced Features

### API Usage
You can also use the API directly:

```bash
# Quick score check
curl -X POST http://localhost:5000/api/quick-score \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "your resume text", "job_description": "job description text"}'

# Keyword analysis
curl -X POST http://localhost:5000/api/keywords \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "your resume text", "job_description": "job description text"}'
```

### Batch Processing
For multiple resumes, you can create a script using the API endpoints.

## 🐛 Troubleshooting

### Common Issues

**"No file selected" Error**
- Ensure you've selected a file before clicking analyze
- Check that file format is supported (PDF, TXT, DOC, DOCX)

**"File too large" Error**
- Reduce file size to under 16MB
- Try saving as a different format

**Import Errors**
- Run: `pip install -r requirements_ats.txt`
- Ensure Python 3.8+ is installed

**Low Scores Despite Good Resume**
- Job description might be too generic
- Try using a more detailed job posting
- Ensure resume contains relevant keywords

### Performance Tips
- Use specific, detailed job descriptions for better analysis
- Include complete resume content (not just summary)
- Ensure good internet connection for initial NLTK downloads

## 📈 Improving Your Score

### Immediate Actions (Can improve score by 10-20%)
1. Add 5-10 missing keywords naturally throughout resume
2. Use exact job title if you qualify
3. Include required years of experience
4. Add relevant technical skills

### Medium-term Actions (Can improve score by 20-40%)
1. Restructure resume with standard sections
2. Quantify all achievements with numbers
3. Add relevant certifications or courses
4. Include industry-specific terminology

### Long-term Actions (Can improve score by 40%+)
1. Gain experience in missing skill areas
2. Obtain relevant certifications
3. Build projects showcasing required technologies
4. Network within target industry

## 🎯 Industry-Specific Tips

### Technology
- Focus on programming languages and frameworks
- Include cloud platforms (AWS, Azure, GCP)
- Mention development methodologies (Agile, Scrum)
- Quantify code contributions and system improvements

### Healthcare
- Emphasize certifications and licenses
- Include patient care metrics
- Mention compliance and regulatory knowledge
- Highlight continuing education

### Finance
- Focus on analytical skills and tools
- Include relevant certifications (CFA, CPA)
- Mention regulatory knowledge
- Quantify financial impact of your work

### Marketing
- Emphasize campaign results and metrics
- Include digital marketing tools and platforms
- Mention brand management experience
- Quantify audience growth and engagement

## 🔄 Iterative Improvement

1. **Initial Analysis**: Get baseline score
2. **Quick Wins**: Add missing keywords and fix formatting
3. **Re-analyze**: Check improved score
4. **Content Updates**: Revise experience descriptions
5. **Final Check**: Ensure 70%+ score for competitive applications

## 📞 Support

If you encounter issues:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure file formats are supported
4. Try with a smaller file if upload fails

---

**Remember**: ATS optimization is just the first step. Once you pass the ATS, your resume still needs to impress human recruiters!