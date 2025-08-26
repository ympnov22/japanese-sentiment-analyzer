# Japanese Text Sentiment Analysis Web App Specification

## 1. App Purpose

Provide a web application that receives Japanese text as input and classifies it into three categories: "Positive", "Negative", and "Neutral" using machine learning models. Users can easily determine the sentiment polarity of Japanese sentences and check confidence scores.

## 2. Functional Requirements

### 2.1 Basic Features
- **Text Input**: Text box for users to input Japanese sentences
- **Sentiment Analysis**: Execute 3-class classification with "Analyze" button
- **Result Display**: 
  - Classification result (Positive / Negative / Neutral)
  - Confidence score (0-1 range)
- **No Authentication**: Available without login functionality

### 2.2 Non-functional Requirements
- **Response Time**: Classification processing within 3 seconds
- **Availability**: 24/7 operation capability
- **Scalability**: Support for dozens of concurrent users

## 3. Data Usage

### 3.1 Training Dataset
- **Data Source**: Hugging Face `daigo/amazon-japanese-reviews` dataset
- **Data URL**: https://huggingface.co/datasets/daigo/amazon-japanese-reviews
- **Sample Size**: 10,000 records (limited for faster training)
- **Data Structure**: 
  - Text: Japanese review text
  - Label: Rating score (1-5) → converted to 3 classes

### 3.2 Label Conversion Rules
- **Positive**: Rating score 4-5
- **Neutral**: Rating score 3
- **Negative**: Rating score 1-2

## 4. Model Configuration

### 4.1 Machine Learning Pipeline
1. **Preprocessing**: Text cleaning, normalization
2. **Feature Extraction**: TF-IDF vectorization
3. **Classifier**: Logistic regression
4. **Post-processing**: Calculate confidence score from probability values

### 4.2 Model Management
- **Storage Location**: `models/` directory
- **File Structure**:
  - `sentiment_model.pkl`: Trained logistic regression model
  - `tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer
  - `model_metadata.json`: Model information (training date, accuracy, etc.)
- **Replaceable Design**: Model path configurable via settings file

## 5. API Specification

### 5.1 Sentiment Analysis Endpoint

**Endpoint**: `POST /predict`

**Request Format**:
```json
{
  "text": "この映画は最高に面白かった！"
}
```

**Response Format**:
```json
{
  "result": "ポジティブ",
  "score": 0.93
}
```

**Error Response**:
```json
{
  "error": "エラーメッセージ",
  "code": "ERROR_CODE"
}
```

### 5.2 Health Check Endpoint

**Endpoint**: `GET /health`

**Response Format**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-08-23T06:23:50Z"
}
```

## 6. UI Configuration

### 6.1 Screen Layout
- **Header**: App title "日本語感情分析"
- **Main Area**:
  - Text input field (multi-line support)
  - Analyze button
  - Result display area
- **Footer**: Technology information

### 6.2 UI Element Details
- **Input Field**: 
  - Placeholder: "分析したい日本語テキストを入力してください"
  - Maximum characters: 1000
- **Analyze Button**: 
  - Label: "感情を判定"
  - Disabled during processing + loading indicator
- **Result Display**:
  - Large display of classification result
  - Confidence visualization with progress bar
  - Color coding (Positive: green, Negative: red, Neutral: gray)

### 6.3 Styling Approach
- **Initial Implementation**: Minimal styling with basic color coding and layout
- **Enhancement**: Additional styling improvements can be added as separate tasks if needed

## 7. Technology Stack

### 7.1 Backend
- **Language**: Python 3.12
- **Framework**: FastAPI
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas
- **Dataset**: datasets (Hugging Face)
- **Server**: uvicorn

### 7.2 Frontend
- **Markup**: HTML5
- **Styling**: CSS3
- **Script**: JavaScript (ES6+)
- **UI**: Responsive design

### 7.3 Development & Operations
- **Dependency Management**: Poetry
- **API Documentation**: FastAPI auto-generation
- **Logging**: Python standard logging
- **Configuration Management**: Environment variables + .env

## 8. Directory Structure

```
japanese-sentiment-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models/              # Data model definitions
│   │   ├── services/            # Business logic
│   │   └── utils/               # Utility functions
│   ├── models/                  # Trained model storage
│   ├── scripts/                 # Training scripts
│   ├── pyproject.toml           # Dependency definitions
│   └── .env                     # Environment variables
├── frontend/
│   ├── index.html               # Main page
│   ├── style.css                # Stylesheet
│   ├── script.js                # JavaScript
│   └── .env                     # Frontend configuration
├── README.md                    # Setup instructions
├── test_predict.py              # Testing script
└── docs/specs/specification.md  # This document
```

## 9. Quality Assurance

### 9.1 Testing Strategy
- **Unit Tests**: Function-level testing
- **Integration Tests**: API endpoint testing
- **E2E Tests**: Frontend to backend flow testing

### 9.2 Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-specific precision
- **Recall**: Class-specific recall
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance across all classes

## 10. Constraints

- Japanese text only
- Training data limited to Amazon reviews (domain dependency)
- No real-time learning functionality
- No user data persistence
- Limited concurrent connections (scalability constraints)

## 11. Development Process

### 11.1 Phase-by-Phase Review
- Each development phase requires completion report and approval before proceeding
- Progress reports include: completed tasks, deliverables, issues encountered, next steps
- Implementation proceeds only after user review and approval of each phase

### 11.2 Deployment Options
- **Primary**: Fly.io (FastAPI backend deployment)
- **Alternatives**: Heroku, Vercel, Render (configurable based on user preference)
- **Frontend**: Static site deployment compatible with multiple platforms

## 12. Future Enhancement Possibilities

- Training with other Japanese datasets
- Advanced models (BERT, etc.)
- Batch processing functionality
- User feedback functionality
- Analysis history storage functionality
- Enhanced UI/UX styling and animations
