const CONFIG = {
    API_BASE_URL: '', // Will be loaded from environment
    MAX_TEXT_LENGTH: 1000,
    DEBUG: true // Set to false for production
};

const elements = {
    textInput: null,
    charCount: null,
    inputError: null,
    analyzeBtn: null,
    buttonText: null,
    loadingSpinner: null,
    resultSection: null,
    sentimentLabel: null,
    confidenceScore: null,
    confidenceFill: null,
    errorSection: null,
    errorMessage: null,
    retryBtn: null,
    apiStatus: null
};

const state = {
    isLoading: false,
    apiHealthy: false
};

function log(message, ...args) {
    if (CONFIG.DEBUG) {
        console.log(`[SentimentAnalyzer] ${message}`, ...args);
    }
}

function logError(message, error) {
    console.error(`[SentimentAnalyzer] ${message}`, error);
}

async function loadConfig() {
    try {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            CONFIG.API_BASE_URL = 'http://localhost:8000';
        } else {
            CONFIG.API_BASE_URL = 'https://jpn-sentiment-api-nrt.fly.dev';
        }
        log('Configuration loaded', CONFIG);
    } catch (error) {
        logError('Failed to load configuration', error);
        CONFIG.API_BASE_URL = 'http://localhost:8000'; // Fallback
    }
}

function initializeElements() {
    elements.textInput = document.getElementById('text-input');
    elements.charCount = document.getElementById('char-count');
    elements.inputError = document.getElementById('input-error');
    elements.analyzeBtn = document.getElementById('analyze-btn');
    elements.buttonText = elements.analyzeBtn.querySelector('.button-text');
    elements.loadingSpinner = elements.analyzeBtn.querySelector('.loading-spinner');
    elements.resultSection = document.getElementById('result-section');
    elements.sentimentLabel = document.getElementById('sentiment-label');
    elements.confidenceScore = document.getElementById('confidence-score');
    elements.confidenceFill = document.getElementById('confidence-fill');
    elements.errorSection = document.getElementById('error-section');
    elements.errorMessage = document.getElementById('error-message');
    elements.retryBtn = document.getElementById('retry-btn');
    elements.apiStatus = document.getElementById('api-status');
    
    addInputAnimations();
    
    log('DOM elements initialized');
}

function addInputAnimations() {
    const textInput = elements.textInput;
    
    textInput.addEventListener('focus', () => {
        textInput.parentElement.classList.add('focused');
    });
    
    textInput.addEventListener('blur', () => {
        textInput.parentElement.classList.remove('focused');
    });
    
    textInput.addEventListener('input', (e) => {
        if (e.target.value.length > 0) {
            textInput.parentElement.classList.add('has-content');
        } else {
            textInput.parentElement.classList.remove('has-content');
        }
    });
}

function setupEventListeners() {
    elements.textInput.addEventListener('input', handleTextInput);
    elements.textInput.addEventListener('keydown', handleKeyDown);
    
    elements.analyzeBtn.addEventListener('click', handleAnalyzeClick);
    elements.retryBtn.addEventListener('click', handleRetryClick);
    
    log('Event listeners setup complete');
}

function handleTextInput(event) {
    const text = event.target.value;
    const length = text.length;
    
    elements.charCount.textContent = length;
    
    const inputInfo = elements.charCount.parentElement;
    inputInfo.classList.remove('warning', 'error');
    
    if (length > CONFIG.MAX_TEXT_LENGTH * 0.9) {
        inputInfo.classList.add('warning');
    }
    if (length > CONFIG.MAX_TEXT_LENGTH) {
        inputInfo.classList.add('error');
    }
    
    hideInputError();
    
    updateAnalyzeButton();
}

function handleKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (!state.isLoading && isValidInput()) {
            handleAnalyzeClick();
        }
    }
}

async function handleAnalyzeClick() {
    if (state.isLoading) return;
    
    const text = elements.textInput.value.trim();
    
    if (!isValidInput(text)) {
        return;
    }
    
    await analyzeSentiment(text);
}

function handleRetryClick() {
    hideError();
    const text = elements.textInput.value.trim();
    if (isValidInput(text)) {
        analyzeSentiment(text);
    }
}

function isValidInput(text = null) {
    if (text === null) {
        text = elements.textInput.value.trim();
    }
    
    if (!text) {
        showInputError('テキストを入力してください');
        return false;
    }
    
    if (text.length > CONFIG.MAX_TEXT_LENGTH) {
        showInputError(`テキストは${CONFIG.MAX_TEXT_LENGTH}文字以内で入力してください`);
        return false;
    }
    
    return true;
}

function showInputError(message) {
    elements.inputError.textContent = message;
    elements.inputError.classList.remove('hidden');
    elements.textInput.classList.add('error');
}

function hideInputError() {
    elements.inputError.classList.add('hidden');
    elements.textInput.classList.remove('error');
}

function updateAnalyzeButton() {
    const text = elements.textInput.value.trim();
    const isValid = text && text.length <= CONFIG.MAX_TEXT_LENGTH;
    
    elements.analyzeBtn.disabled = !isValid || state.isLoading;
}

function setLoadingState(loading) {
    state.isLoading = loading;
    
    if (loading) {
        elements.buttonText.classList.add('hidden');
        elements.loadingSpinner.classList.remove('hidden');
        elements.analyzeBtn.disabled = true;
    } else {
        elements.buttonText.classList.remove('hidden');
        elements.loadingSpinner.classList.add('hidden');
        updateAnalyzeButton();
    }
}

async function checkApiHealth() {
    try {
        log('Checking API health...');
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        log('API health check response:', data);
        
        state.apiHealthy = data.model_loaded === true;
        
        if (!state.apiHealthy) {
            showApiWarning('APIは動作していますが、モデルが読み込まれていません');
        } else {
            hideApiStatus();
        }
        
        return state.apiHealthy;
        
    } catch (error) {
        logError('API health check failed', error);
        state.apiHealthy = false;
        showApiError('APIサーバーに接続できません。サーバーが起動しているか確認してください。');
        return false;
    }
}

async function analyzeSentiment(text) {
    setLoadingState(true);
    hideResult();
    hideError();
    
    try {
        log('Analyzing sentiment for text:', text.substring(0, 50) + '...');
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        log('Sentiment analysis response:', data);
        
        showResult(data.result, data.score);
        
    } catch (error) {
        logError('Sentiment analysis failed', error);
        showError(getErrorMessage(error));
    } finally {
        setLoadingState(false);
    }
}

function getErrorMessage(error) {
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        return 'ネットワークエラーが発生しました。インターネット接続を確認してください。';
    } else if (error.message.includes('400')) {
        return '入力されたテキストに問題があります。内容を確認してください。';
    } else if (error.message.includes('500')) {
        return 'サーバーエラーが発生しました。しばらく時間をおいて再試行してください。';
    } else if (error.message.includes('503')) {
        return 'サービスが一時的に利用できません。しばらく時間をおいて再試行してください。';
    } else {
        return 'エラーが発生しました。再試行してください。';
    }
}

function showApiWarning(message) {
    elements.apiStatus.textContent = message;
    elements.apiStatus.className = 'api-status warning';
    elements.apiStatus.classList.remove('hidden');
}

function showApiError(message) {
    elements.apiStatus.textContent = message;
    elements.apiStatus.className = 'api-status error';
    elements.apiStatus.classList.remove('hidden');
}

function hideApiStatus() {
    elements.apiStatus.classList.add('hidden');
}

function showResult(sentiment, confidence) {
    elements.sentimentLabel.textContent = sentiment;
    elements.sentimentLabel.className = 'sentiment-label';
    
    const sentimentIcon = document.getElementById('sentiment-icon');
    sentimentIcon.className = 'sentiment-icon';
    
    if (sentiment === 'ポジティブ') {
        elements.sentimentLabel.textContent = 'Positive';
        elements.sentimentLabel.classList.add('positive');
        elements.confidenceFill.className = 'confidence-fill positive';
        sentimentIcon.classList.add('positive');
    } else {
        elements.sentimentLabel.textContent = 'Negative';
        elements.sentimentLabel.classList.add('negative');
        elements.confidenceFill.className = 'confidence-fill negative';
        sentimentIcon.classList.add('negative');
    }
    
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceScore.textContent = `信頼度: ${confidencePercent}%`;
    
    elements.resultSection.classList.remove('hidden');
    
    setTimeout(() => {
        elements.confidenceFill.style.width = `${confidencePercent}%`;
        createConfidenceParticles(confidencePercent);
    }, 300);
    
    elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function createConfidenceParticles(confidence) {
    const particlesContainer = document.getElementById('confidence-particles');
    particlesContainer.innerHTML = '';
    
    const particleCount = Math.floor(confidence / 10);
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particle-float 3s ease-in-out infinite;
            animation-delay: ${Math.random() * 2}s;
        `;
        particlesContainer.appendChild(particle);
    }
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes particle-float {
            0%, 100% { transform: translateY(0px) scale(1); opacity: 0.8; }
            50% { transform: translateY(-10px) scale(1.2); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
}

function hideResult() {
    elements.resultSection.classList.add('hidden');
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorSection.classList.remove('hidden');
    
    elements.errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    elements.errorSection.classList.add('hidden');
}

async function initializeApp() {
    log('Initializing Japanese Sentiment Analyzer...');
    
    await loadConfig();
    initializeElements();
    setupEventListeners();
    
    updateAnalyzeButton();
    
    await checkApiHealth();
    
    log('Application initialized successfully');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
