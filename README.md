# ATS Resume Analyzer

A comprehensive Applicant Tracking System (ATS) resume analyzer built with Flask and Machine Learning. This application helps job seekers optimize their resumes to pass ATS screening and increase their chances of landing interviews.

## 🌟 Features

- **Resume Analysis**: Upload PDF, TXT, DOC, or DOCX files for analysis
- **ATS Scoring**: Get detailed scores based on keyword matching, semantic similarity, and formatting
- **Keyword Analysis**: See which keywords are matched and which are missing
- **Industry-Specific Analysis**: Tailored analysis based on detected industry
- **Improvement Recommendations**: Get actionable suggestions to improve your resume
- **Modern UI**: Clean, responsive web interface with real-time results
- **Secure Processing**: All analysis happens locally on your server

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project files**

2. **Run the setup script**:
   ```bash
   python setup_ats.py
   ```

3. **Configure environment (optional)**:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

4. **Start the application**:
   ```bash
   python app_ats.py
   ```

5. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
ATS Project/
├── app_ats.py                 # Main Flask application
├── ats_service.py             # ATS analysis service
├── ats_scoring.py             # Advanced scoring algorithms
├── requirements_ats.txt       # Python dependencies
├── setup_ats.py              # Automated setup script
├── README_ATS.md              # This file
├── sample backend/            # Original sample code (reference)
├── templates/
│   └── index.html            # Main web interface
├── static/
│   ├── css/
│   │   └── style.css         # Custom styles
│   └── js/
│       └── app.js            # Frontend JavaScript
└── uploads/                   # Temporary file storage
```

## 🔧 API Endpoints

### Main Analysis Endpoint
- **POST** `/api/analyze`
  - Upload resume file and job description
  - Returns comprehensive ATS analysis

### Quick Score Endpoint
- **POST** `/api/quick-score`
  - JSON input with resume text and job description
  - Returns basic ATS score

### Keywords Analysis
- **POST** `/api/keywords`
  - Detailed keyword matching analysis

### Suggestions
- **POST** `/api/suggestions`
  - Get personalized improvement recommendations

## 📊 Analysis Components

### Overall ATS Score
- Weighted combination of multiple factors
- Scale: 0-100%
- Categories: Excellent (80+), Good (60-79), Fair (40-59), Poor (<40)

### Detailed Metrics
1. **Keyword Matching**: Relevance to job description keywords
2. **Semantic Similarity**: Content similarity using NLP
3. **Skills Match**: Technical and soft skills alignment
4. **Experience Match**: Experience level and relevance
5. **Education Match**: Educational background alignment

### Additional Analysis
- **Readability Score**: Flesch reading ease score
- **Formatting Analysis**: ATS-friendly formatting check
- **Industry Detection**: Automatic industry classification
- **Contact Information**: Extraction of contact details

## 🎯 ATS Optimization Tips

### ✅ Do's
- Use standard section headers (Experience, Education, Skills)
- Include relevant keywords from job descriptions
- Use simple, clean formatting
- Save in PDF and plain text formats
- Quantify achievements with numbers
- Use action verbs (managed, developed, implemented)

### ❌ Don'ts
- Avoid graphics, images, or charts
- Don't use tables or complex formatting
- Avoid unusual fonts or colors
- Don't use headers/footers
- Avoid acronyms without explanation
- Don't keyword stuff

## 🔍 How It Works

1. **File Processing**: Extracts text from uploaded resume files
2. **Text Preprocessing**: Cleans and normalizes text data
3. **Keyword Extraction**: Identifies relevant keywords and skills
4. **Similarity Analysis**: Uses TF-IDF and cosine similarity
5. **Industry Detection**: Classifies based on job description content
6. **Scoring Algorithm**: Weighted combination of multiple factors
7. **Recommendation Engine**: Generates personalized improvement tips

## 🛠️ Technical Details

### Backend Technologies
- **Flask**: Web framework
- **scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **PDFMiner**: PDF text extraction
- **NumPy/Pandas**: Data processing

### Frontend Technologies
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons
- **Vanilla JavaScript**: Interactive functionality
- **CSS3**: Custom styling and animations

### Machine Learning Features
- **TF-IDF Vectorization**: Text feature extraction
- **Cosine Similarity**: Semantic similarity measurement
- **Keyword Matching**: Rule-based keyword analysis
- **Text Preprocessing**: Stemming, stopword removal
- **Industry Classification**: Keyword-based industry detection

## 🔒 Security & Privacy

- All processing happens locally on your server
- No data is sent to external services (unless configured)
- Uploaded files are temporarily stored and can be automatically cleaned
- No personal information is logged or stored permanently

## 📈 Performance

- Average analysis time: 2-5 seconds
- Supports files up to 16MB
- Concurrent request handling
- Optimized for production deployment

## 🚀 Deployment

### Development
```bash
python app_ats.py
```

### Production (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app_ats:app
```

### Docker (optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_ats.txt .
RUN pip install -r requirements_ats.txt
COPY . .
EXPOSE 5000
CMD ["python", "app_ats.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check the console logs for error messages
2. Ensure all dependencies are installed correctly
3. Verify file formats are supported (PDF, TXT, DOC, DOCX)
4. Check file size limits (16MB max)

## 🔮 Future Enhancements

- [ ] Support for more file formats
- [ ] Integration with job boards APIs
- [ ] Resume template suggestions
- [ ] Batch processing capabilities
- [ ] Advanced NLP models (BERT, GPT)
- [ ] Multi-language support
- [ ] Resume builder integration
- [ ] A/B testing for resume versions

---

**Built with ❤️ using Flask, Machine Learning, and modern web technologies.**