const CONFIG = {
    API_BASE_URL: '', // Will be loaded from environment
    MAX_TEXT_LENGTH: 1000,
    DEBUG: true, // Set to false for production
    HISTORY_MAX_ITEMS: 5
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
    apiStatus: null,
    themeToggle: null,
    sunIcon: null,
    moonIcon: null,
    historySection: null,
    historyList: null,
    analysisTime: null
};

const state = {
    isLoading: false,
    apiHealthy: false,
    currentTheme: 'light',
    analysisHistory: []
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
        } else if (window.location.hostname === 'japanese-sentiment-frontend-staging.fly.dev') {
            CONFIG.API_BASE_URL = 'https://japanese-sentiment-analyzer-staging.fly.dev';
        } else {
            CONFIG.API_BASE_URL = 'https://japanese-sentiment-analyzer.fly.dev';
        }
        log('Configuration loaded', CONFIG);
        
        if (CONFIG.API_BASE_URL.includes('staging')) {
            console.log(`[STAGING] API Base URL: ${CONFIG.API_BASE_URL}`);
        }
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
    elements.buttonText = document.getElementById('button-text');
    elements.loadingSpinner = document.getElementById('loading-spinner');
    elements.resultSection = document.getElementById('result-section');
    elements.sentimentLabel = document.getElementById('sentiment-label');
    elements.confidenceScore = document.getElementById('confidence-score');
    elements.confidenceFill = document.getElementById('confidence-fill');
    elements.errorSection = document.getElementById('error-section');
    elements.errorMessage = document.getElementById('error-message');
    elements.retryBtn = document.getElementById('retry-btn');
    elements.apiStatus = document.getElementById('api-status');
    elements.themeToggle = document.getElementById('theme-toggle');
    elements.sunIcon = document.getElementById('sun-icon');
    elements.moonIcon = document.getElementById('moon-icon');
    elements.historySection = document.getElementById('history-section');
    elements.historyList = document.getElementById('history-list');
    elements.analysisTime = document.getElementById('analysis-time');
    
    log('DOM elements initialized');
}

function setupEventListeners() {
    elements.textInput.addEventListener('input', handleTextInput);
    elements.textInput.addEventListener('keydown', handleKeyDown);
    
    elements.analyzeBtn.addEventListener('click', handleAnalyzeClick);
    elements.retryBtn.addEventListener('click', handleRetryClick);
    elements.themeToggle.addEventListener('click', handleThemeToggle);
    
    log('Event listeners setup complete');
}

function handleTextInput(event) {
    const text = event.target.value;
    const length = text.length;
    
    elements.charCount.textContent = length;
    
    const charCountElement = elements.charCount.parentElement;
    charCountElement.classList.remove('char-count', 'warning', 'error');
    charCountElement.classList.add('char-count');
    
    if (length > CONFIG.MAX_TEXT_LENGTH * 0.9) {
        charCountElement.classList.add('warning');
    }
    if (length > CONFIG.MAX_TEXT_LENGTH) {
        charCountElement.classList.add('error');
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
        elements.buttonText.textContent = '分析中...';
        elements.loadingSpinner.classList.remove('hidden');
        elements.analyzeBtn.disabled = true;
    } else {
        elements.buttonText.textContent = '感情分析';
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
    elements.sentimentLabel.className = 'inline-flex items-center px-4 py-2 rounded-lg font-semibold text-lg border';
    
    if (sentiment === 'ポジティブ') {
        elements.sentimentLabel.classList.add('sentiment-positive');
        elements.confidenceFill.className = 'h-3 rounded-full transition-all duration-500 ease-out confidence-positive';
    } else {
        elements.sentimentLabel.classList.add('sentiment-negative');
        elements.confidenceFill.className = 'h-3 rounded-full transition-all duration-500 ease-out confidence-negative';
    }
    
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceScore.textContent = `信頼度: ${confidencePercent}%`;
    
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    
    const now = new Date();
    elements.analysisTime.textContent = `分析時刻: ${now.toLocaleString('ja-JP')}`;
    
    elements.resultSection.classList.remove('hidden');
    
    addToHistory(elements.textInput.value.trim(), sentiment, confidence, now);
    
    elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    initializeTheme();
    loadHistory();
    
    updateAnalyzeButton();
    
    await checkApiHealth();
    
    log('Application initialized successfully');
}

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        state.currentTheme = savedTheme;
    } else {
        state.currentTheme = prefersDark ? 'dark' : 'light';
    }
    
    applyTheme(state.currentTheme);
    
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            state.currentTheme = e.matches ? 'dark' : 'light';
            applyTheme(state.currentTheme);
        }
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    state.currentTheme = theme;
    
    if (theme === 'dark') {
        elements.sunIcon.classList.remove('hidden');
        elements.moonIcon.classList.add('hidden');
    } else {
        elements.sunIcon.classList.add('hidden');
        elements.moonIcon.classList.remove('hidden');
    }
}

function handleThemeToggle() {
    const newTheme = state.currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    log(`Theme switched to: ${newTheme}`);
}

function addToHistory(text, sentiment, confidence, timestamp) {
    const historyItem = {
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        fullText: text,
        sentiment,
        confidence,
        timestamp
    };
    
    state.analysisHistory.unshift(historyItem);
    
    if (state.analysisHistory.length > CONFIG.HISTORY_MAX_ITEMS) {
        state.analysisHistory = state.analysisHistory.slice(0, CONFIG.HISTORY_MAX_ITEMS);
    }
    
    saveHistory();
    renderHistory();
}

function saveHistory() {
    try {
        localStorage.setItem('analysisHistory', JSON.stringify(state.analysisHistory));
    } catch (error) {
        logError('Failed to save history', error);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('analysisHistory');
        if (saved) {
            state.analysisHistory = JSON.parse(saved);
            renderHistory();
        }
    } catch (error) {
        logError('Failed to load history', error);
        state.analysisHistory = [];
    }
}

function renderHistory() {
    if (state.analysisHistory.length === 0) {
        elements.historyList.innerHTML = '<p class="text-slate-500 dark:text-slate-400 text-sm">まだ分析履歴がありません</p>';
        return;
    }
    
    const historyHTML = state.analysisHistory.map((item, index) => {
        const confidencePercent = Math.round(item.confidence * 100);
        const sentimentClass = item.sentiment === 'ポジティブ' ? 'sentiment-positive' : 'sentiment-negative';
        const timeStr = new Date(item.timestamp).toLocaleString('ja-JP');
        
        return `
            <div class="history-item" tabindex="0" role="button" data-index="${index}" aria-label="履歴項目: ${item.text}">
                <div class="flex justify-between items-start gap-3">
                    <div class="flex-1 min-w-0">
                        <p class="text-sm text-slate-700 dark:text-slate-300 truncate">${item.text}</p>
                        <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">${timeStr}</p>
                    </div>
                    <div class="flex items-center gap-2 flex-shrink-0">
                        <span class="inline-flex items-center px-2 py-1 rounded text-xs font-medium border ${sentimentClass}">${item.sentiment}</span>
                        <span class="text-xs text-slate-500 dark:text-slate-400">${confidencePercent}%</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    elements.historyList.innerHTML = historyHTML;
    
    elements.historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', handleHistoryItemClick);
        item.addEventListener('keydown', handleHistoryItemKeydown);
    });
}

function handleHistoryItemClick(event) {
    const index = parseInt(event.currentTarget.dataset.index);
    const historyItem = state.analysisHistory[index];
    if (historyItem) {
        elements.textInput.value = historyItem.fullText;
        elements.textInput.focus();
        handleTextInput({ target: elements.textInput });
        
        elements.textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function handleHistoryItemKeydown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        handleHistoryItemClick(event);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
