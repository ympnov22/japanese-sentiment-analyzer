# Japanese Sentiment Analysis Web App

A web application that classifies Japanese text into "Positive," "Negative," or "Neutral" categories using machine learning.

## Features

- **Text Input**: Multi-line text box for Japanese sentence input
- **Sentiment Analysis**: 3-class classification with confidence scores
- **Real-time Results**: Instant sentiment analysis with visual feedback
- **No Authentication**: Login-free usage
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

### Backend
- **Python 3.12** with **FastAPI**
- **scikit-learn** for machine learning (TF-IDF + Logistic Regression)
- **pandas** for data processing
- **Hugging Face datasets** for training data
- **Poetry** for dependency management

### Frontend
- **HTML5** + **CSS3** + **JavaScript**
- Responsive design with minimal styling
- RESTful API integration

### Data
- **Training Dataset**: Hugging Face `daigo/amazon-japanese-reviews`
- **Sample Size**: 10,000 records for fast training
- **Label Conversion**: 5-point rating → 3-class sentiment

## Project Structure

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
│   ├── pyproject.toml           # Dependencies
│   └── .env                     # Environment variables
├── frontend/
│   ├── index.html               # Main page
│   ├── style.css                # Stylesheet
│   ├── script.js                # JavaScript
│   └── .env                     # Frontend configuration
├── docs/                        # Documentation (see docs/README.md)
│   ├── specs/                   # Technical specifications
│   ├── reports/                 # Current reports and analysis
│   ├── archive/                 # Historical development documentation
│   └── releases/                # Release notes and changelog
├── README.md                    # This file (see docs/README.md for full docs)
```

## Setup Instructions

### Prerequisites
- Python 3.12+
- Poetry (for dependency management)
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ympnov22/japanese-sentiment-analyzer.git
   cd japanese-sentiment-analyzer
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   poetry install
   ```

3. **Configure environment variables**
   ```bash
   # Edit backend/.env file with your settings
   cp .env.example .env
   ```

4. **Train the model** (Phase 2+)
   ```bash
   poetry run python scripts/train_model.py
   ```

5. **Start the development server**
   ```bash
   poetry run fastapi dev app/main.py
   ```

   The API will be available at: http://localhost:8000
   API documentation: http://localhost:8000/docs

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Configure API URL (optional)**
   ```bash
   # The frontend automatically uses http://localhost:8000 for local development
   # Edit frontend/.env file only if you need a different API URL
   ```

3. **Start the frontend**
   
   **Option 1: Direct file access**
   ```bash
   # Open index.html directly in your browser
   open index.html  # macOS
   start index.html  # Windows
   xdg-open index.html  # Linux
   ```
   
   **Option 2: HTTP Server (recommended)**
   ```bash
   # Using Python's built-in server
   python -m http.server 3000
   
   # Or using Node.js serve
   npx serve . -p 3000
   ```
   
   The app will be available at: http://localhost:3000

4. **Usage**
   - Enter Japanese text in the textarea (max 1000 characters)
   - Click "感情を判定" (Analyze Sentiment) button
   - View results with sentiment classification and confidence score
   - The interface is responsive and works on mobile devices

## API Endpoints

### POST /predict
Analyze sentiment of Japanese text.

**Request:**
```json
{
  "text": "この映画は最高に面白かった！"
}
```

**Response:**
```json
{
  "result": "ポジティブ",
  "score": 0.93
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-08-23T08:23:29Z"
}
```

## Development Workflow

### Phase-by-Phase Implementation
1. **Phase 1**: Environment Setup ✅
2. **Phase 2**: Data Preparation & Preprocessing
3. **Phase 3**: ML Model Development
4. **Phase 4**: Backend API Development
5. **Phase 5**: Frontend Development
6. **Phase 6**: Testing & QA
7. **Phase 7**: Documentation & Deployment

### Testing

**Run unit tests:**
```bash
cd backend
poetry run pytest
```

**Test API endpoints:**
```bash
python test_predict.py
```

**Manual testing:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "素晴らしい映画でした"}'
```

## デプロイ

### ローカル開発環境

#### バックエンド起動
```bash
cd backend
poetry install
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### フロントエンド起動
```bash
cd frontend
# 静的ファイルサーバーで配信
python -m http.server 8080
```

### 本番環境デプロイ (Fly.io)

#### 前提条件
- Fly.ioアカウントの作成
- flyctl CLIのインストール
- FLY_API_TOKEN (org deploy token) の取得

#### バックエンドデプロイ
```bash
cd backend
flyctl deploy --config fly.toml
```

#### フロントエンドデプロイ
```bash
cd frontend
flyctl deploy --config fly.toml
```

#### デプロイ後の確認
```bash
# API健康チェック
curl https://jpn-sentiment-api-nrt.fly.dev/health

# Webアプリアクセス
# https://jpn-sentiment-web-nrt.fly.dev
```

#### ログ確認
```bash
# APIログ
flyctl logs --app jpn-sentiment-api-nrt

# Webアプリログ
flyctl logs --app jpn-sentiment-web-nrt
```

#### ロールバック
```bash
# 前のバージョンにロールバック
flyctl releases --app jpn-sentiment-api-nrt
flyctl rollback --app jpn-sentiment-api-nrt <release-id>

flyctl releases --app jpn-sentiment-web-nrt
flyctl rollback --app jpn-sentiment-web-nrt <release-id>
```

#### アプリケーション管理
```bash
# アプリ状態確認
flyctl status --app jpn-sentiment-api-nrt
flyctl status --app jpn-sentiment-web-nrt

# アプリ停止/開始
flyctl scale count 0 --app jpn-sentiment-api-nrt  # 停止
flyctl scale count 1 --app jpn-sentiment-api-nrt  # 開始
```

## Deployment Options

The application supports deployment to multiple platforms:
- **Fly.io** (recommended for FastAPI)
- **Heroku**
- **Vercel** (frontend)
- **Render**

## Model Information

- **Algorithm**: TF-IDF Vectorization + Logistic Regression
- **Training Data**: Japanese Sentiment Dataset (9,762 samples)
- **Classes**: Positive (ポジティブ), Negative (ネガティブ) - Binary Classification
- **Performance**: 94.5% training accuracy, 90.4% test accuracy
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Frontend Features

- **Responsive Design**: Works on desktop and mobile (375px+ width)
- **Input Validation**: Real-time character counting and validation
- **Error Handling**: User-friendly error messages for network issues
- **Loading States**: Visual feedback during API requests
- **Accessibility**: Keyboard shortcuts (Ctrl/Cmd + Enter to analyze)
- **Visual Feedback**: Color-coded results (green=positive, red=negative)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [daigo/amazon-japanese-reviews](https://huggingface.co/datasets/daigo/amazon-japanese-reviews) from Hugging Face
- **Framework**: FastAPI for high-performance web APIs
- **ML Library**: scikit-learn for machine learning capabilities

## Contact

- **Developer**: Devin AI
- **User**: ヤマシタ　ヤスヒロ (@ympnov22)
- **Repository**: https://github.com/ympnov22/japanese-sentiment-analyzer
