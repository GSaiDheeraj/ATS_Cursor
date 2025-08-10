// ATS Resume Analyzer JavaScript

class ATSAnalyzer {
    constructor() {
        this.currentAnalysis = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('ats-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeResume();
        });

        // File removal
        document.getElementById('remove-file').addEventListener('click', () => {
            this.removeFile();
        });

        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('resume-file');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.handleFileSelect(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
    }

    handleFileSelect(file) {
        const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 
                             'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a PDF, TXT, DOC, or DOCX file.');
            return;
        }

        if (file.size > 16 * 1024 * 1024) { // 16MB
            this.showError('File size must be less than 16MB.');
            return;
        }

        // Show file info
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-info').style.display = 'block';
        document.querySelector('.upload-text').style.display = 'none';
    }

    removeFile() {
        document.getElementById('resume-file').value = '';
        document.getElementById('file-info').style.display = 'none';
        document.querySelector('.upload-text').style.display = 'block';
    }

    async analyzeResume() {
        const form = document.getElementById('ats-form');
        const formData = new FormData(form);

        // Validate form
        if (!formData.get('resume') || !formData.get('resume').name) {
            this.showError('Please select a resume file.');
            return;
        }

        if (!formData.get('job_description').trim()) {
            this.showError('Please enter a job description.');
            return;
        }

        // Show loading
        this.showLoading();

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status) {
                this.currentAnalysis = result;
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        document.getElementById('loading-section').style.display = 'block';
        document.getElementById('results-section').style.display = 'none';
        document.getElementById('analyze-btn').disabled = true;
        
        // Scroll to loading section
        document.getElementById('loading-section').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }

    hideLoading() {
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('analyze-btn').disabled = false;
    }

    displayResults(data) {
        const analysis = data.analysis;
        const additionalMetrics = data.additional_metrics;

        // Update overall score
        this.updateOverallScore(analysis);

        // Update score breakdown
        this.updateScoreBreakdown(analysis.detailed_scores);

        // Update keywords analysis
        this.updateKeywordsAnalysis(analysis.keyword_analysis);

        // Update recommendations
        this.updateRecommendations(analysis.recommendations);

        // Update additional metrics
        this.updateAdditionalMetrics(additionalMetrics);

        // Show results section
        document.getElementById('results-section').style.display = 'block';
        
        // Scroll to results
        document.getElementById('results-section').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    updateOverallScore(analysis) {
        const scoreElement = document.getElementById('overall-score');
        const levelElement = document.getElementById('compatibility-level');
        const statusElement = document.getElementById('status-message');
        const circleElement = document.getElementById('score-circle');

        const score = Math.round(analysis.overall_score);
        
        // Animate score
        this.animateScore(scoreElement, score);
        
        levelElement.textContent = analysis.compatibility_level;
        statusElement.textContent = analysis.status;

        // Update circle color and gradient
        circleElement.className = 'score-circle mx-auto mb-3';
        if (score >= 80) {
            circleElement.classList.add('score-excellent');
        } else if (score >= 60) {
            circleElement.classList.add('score-good');
        } else if (score >= 40) {
            circleElement.classList.add('score-fair');
        } else {
            circleElement.classList.add('score-poor');
        }

        // Update circle gradient
        const percentage = (score / 100) * 360;
        circleElement.style.setProperty('--score-percentage', `${percentage}deg`);
    }

    animateScore(element, targetScore) {
        let currentScore = 0;
        const increment = targetScore / 50;
        const timer = setInterval(() => {
            currentScore += increment;
            if (currentScore >= targetScore) {
                currentScore = targetScore;
                clearInterval(timer);
            }
            element.textContent = Math.round(currentScore);
        }, 30);
    }

    updateScoreBreakdown(scores) {
        const container = document.getElementById('score-breakdown');
        container.innerHTML = '';

        const scoreItems = [
            { label: 'Keyword Matching', value: scores.keyword_matching, color: 'bg-primary' },
            { label: 'Semantic Similarity', value: scores.semantic_similarity, color: 'bg-info' },
            { label: 'Skills Match', value: scores.skills_match, color: 'bg-success' },
            { label: 'Experience Match', value: scores.experience_match, color: 'bg-warning' },
            { label: 'Education Match', value: scores.education_match, color: 'bg-secondary' }
        ];

        scoreItems.forEach(item => {
            const scoreBar = this.createScoreBar(item.label, item.value, item.color);
            container.appendChild(scoreBar);
        });
    }

    createScoreBar(label, value, colorClass) {
        const scoreBar = document.createElement('div');
        scoreBar.className = 'score-bar';
        
        const roundedValue = Math.round(value);
        
        scoreBar.innerHTML = `
            <div class="score-label">
                <span>${label}</span>
                <span><strong>${roundedValue}%</strong></span>
            </div>
            <div class="progress">
                <div class="progress-bar ${colorClass}" role="progressbar" 
                     style="width: ${roundedValue}%" aria-valuenow="${roundedValue}" 
                     aria-valuemin="0" aria-valuemax="100">
                </div>
            </div>
        `;

        return scoreBar;
    }

    updateKeywordsAnalysis(keywordAnalysis) {
        const container = document.getElementById('keywords-analysis');
        
        const matchedCount = keywordAnalysis.matched_keywords.length;
        const missingCount = keywordAnalysis.missing_keywords.length;
        
        container.innerHTML = `
            <div class="mb-3">
                <h6 class="text-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Matched Keywords (${matchedCount})
                </h6>
                <div class="keywords-container">
                    ${keywordAnalysis.matched_keywords.slice(0, 10).map(keyword => 
                        `<span class="keyword-tag keyword-matched">${keyword}</span>`
                    ).join('')}
                    ${matchedCount > 10 ? `<span class="text-muted">... and ${matchedCount - 10} more</span>` : ''}
                </div>
            </div>
            <div class="mb-3">
                <h6 class="text-danger">
                    <i class="fas fa-times-circle me-2"></i>
                    Missing Keywords (${missingCount})
                </h6>
                <div class="keywords-container">
                    ${keywordAnalysis.missing_keywords.slice(0, 10).map(keyword => 
                        `<span class="keyword-tag keyword-missing">${keyword}</span>`
                    ).join('')}
                    ${missingCount > 10 ? `<span class="text-muted">... and ${missingCount - 10} more</span>` : ''}
                </div>
            </div>
            <div class="alert alert-info">
                <strong>Match Rate:</strong> ${keywordAnalysis.match_percentage}%
            </div>
        `;
    }

    updateRecommendations(recommendations) {
        const container = document.getElementById('recommendations');
        
        if (recommendations.length === 0) {
            container.innerHTML = '<p class="text-muted">Great job! No specific recommendations at this time.</p>';
            return;
        }

        container.innerHTML = recommendations.map((rec, index) => `
            <div class="recommendation-item recommendation-medium">
                <div class="d-flex align-items-start">
                    <i class="fas fa-lightbulb text-warning me-3 mt-1"></i>
                    <div>
                        <strong>Tip ${index + 1}:</strong> ${rec}
                    </div>
                </div>
            </div>
        `).join('');
    }

    updateAdditionalMetrics(metrics) {
        // Readability metrics
        const readabilityContainer = document.getElementById('readability-metrics');
        if (metrics.readability) {
            readabilityContainer.innerHTML = `
                <div class="metric-value text-info">${metrics.readability.flesch_score}</div>
                <div class="metric-label">Flesch Score</div>
                <small class="text-muted">Level: ${metrics.readability.readability_level}</small>
            `;
        }

        // Formatting metrics
        const formattingContainer = document.getElementById('formatting-metrics');
        if (metrics.formatting) {
            formattingContainer.innerHTML = `
                <div class="metric-value text-success">${metrics.formatting.formatting_score}%</div>
                <div class="metric-label">ATS Friendly</div>
                <small class="text-muted">
                    ${metrics.formatting.ats_friendly ? 
                        '<i class="fas fa-check text-success"></i> Good formatting' : 
                        '<i class="fas fa-exclamation-triangle text-warning"></i> Needs improvement'
                    }
                </small>
            `;
        }

        // Industry metrics
        const industryContainer = document.getElementById('industry-metrics');
        if (metrics.industry_analysis) {
            industryContainer.innerHTML = `
                <div class="metric-value text-secondary">${Math.round(metrics.industry_analysis.industry_score)}%</div>
                <div class="metric-label">Industry Match</div>
                <small class="text-muted">Detected: ${metrics.industry_analysis.industry_detected}</small>
            `;
        }
    }

    showError(message) {
        // Create or update error alert
        let errorAlert = document.getElementById('error-alert');
        if (!errorAlert) {
            errorAlert = document.createElement('div');
            errorAlert.id = 'error-alert';
            errorAlert.className = 'alert alert-danger alert-dismissible fade show';
            document.querySelector('#analyzer .container').prepend(errorAlert);
        }

        errorAlert.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorAlert) {
                errorAlert.remove();
            }
        }, 5000);
    }
}

// Global functions for action buttons
function downloadReport() {
    if (!window.atsAnalyzer || !window.atsAnalyzer.currentAnalysis) {
        alert('No analysis data available to download.');
        return;
    }

    const data = window.atsAnalyzer.currentAnalysis;
    const reportContent = generateReportContent(data);
    
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ats-analysis-report.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function generateReportContent(data) {
    const analysis = data.analysis;
    const metadata = data.metadata;
    
    return `
ATS RESUME ANALYSIS REPORT
Generated on: ${metadata.analysis_timestamp}

OVERALL SCORE: ${analysis.overall_score}%
COMPATIBILITY LEVEL: ${analysis.compatibility_level}
STATUS: ${analysis.status}

DETAILED SCORES:
- Keyword Matching: ${Math.round(analysis.detailed_scores.keyword_matching)}%
- Semantic Similarity: ${Math.round(analysis.detailed_scores.semantic_similarity)}%
- Skills Match: ${Math.round(analysis.detailed_scores.skills_match)}%
- Experience Match: ${Math.round(analysis.detailed_scores.experience_match)}%
- Education Match: ${Math.round(analysis.detailed_scores.education_match)}%

MATCHED KEYWORDS:
${analysis.keyword_analysis.matched_keywords.join(', ')}

MISSING KEYWORDS (Top 15):
${analysis.keyword_analysis.missing_keywords.slice(0, 15).join(', ')}

RECOMMENDATIONS:
${analysis.recommendations.map((rec, i) => `${i + 1}. ${rec}`).join('\n')}

PROCESSING DETAILS:
- Resume Word Count: ${metadata.resume_word_count}
- Job Description Word Count: ${metadata.jd_word_count}
- Processing Time: ${metadata.processing_time_seconds} seconds
    `.trim();
}

function resetAnalysis() {
    // Reset form
    document.getElementById('ats-form').reset();
    
    // Hide results
    document.getElementById('results-section').style.display = 'none';
    
    // Reset file upload
    document.getElementById('file-info').style.display = 'none';
    document.querySelector('.upload-text').style.display = 'block';
    
    // Clear current analysis
    if (window.atsAnalyzer) {
        window.atsAnalyzer.currentAnalysis = null;
    }
    
    // Scroll to top
    document.getElementById('analyzer').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

function shareResults() {
    if (!window.atsAnalyzer || !window.atsAnalyzer.currentAnalysis) {
        alert('No analysis data available to share.');
        return;
    }

    const analysis = window.atsAnalyzer.currentAnalysis.analysis;
    const shareText = `I just analyzed my resume with ATS Resume Analyzer! 
Overall Score: ${analysis.overall_score}% - ${analysis.compatibility_level}
Check it out: ${window.location.href}`;

    if (navigator.share) {
        navigator.share({
            title: 'ATS Resume Analysis Results',
            text: shareText,
            url: window.location.href
        });
    } else if (navigator.clipboard) {
        navigator.clipboard.writeText(shareText).then(() => {
            alert('Results copied to clipboard!');
        });
    } else {
        // Fallback
        const textArea = document.createElement('textarea');
        textArea.value = shareText;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        alert('Results copied to clipboard!');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.atsAnalyzer = new ATSAnalyzer();
});