# Phase 2 実装計画: モデル改善

## 概要
Phase 1で検出された重大なバイアス問題を解決するためのモデル改善実装

**対象リポジトリ**: ympnov22/japanese-sentiment-analyzer  
**ブランチ**: devin/1756116053-model-accuracy-improvement  
**実装期間**: 2025年8月25日

## Phase 1で検出された問題

### 🚨 重大な問題
- **バイアス検出**: 両サニティテストが「ネガティブ」分類（スコア: 0.500）
- **性能劣化**: Model F1 (0.276) < Baseline F1 (0.383)
- **極端な偏り**: ポジティブクラス正解率 0.1% (1/908件)
- **スコア範囲**: 0.500-0.538（極めて狭い）

### 根本原因分析
1. **データリーク**: 全データでTF-IDFをfitしてから分割
2. **クラス不均衡**: class_weight="balanced"が機能していない
3. **TF-IDFパラメータ**: 日本語テキストに最適化されていない
4. **確率校正**: 予測確率が0.5付近に集中

## Phase 2 実装タスク

### 🎯 優先度1: パイプライン化とリーク防止
**目的**: データリークを完全に防止

**実装内容**:
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2'
    )),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=200,
        n_jobs=-1,
        random_state=42
    ))
])

# StratifiedKFold with Pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 🎯 優先度2: TF-IDF最適化
**推奨初期設定**:
- `analyzer='char'`: 日本語の文字レベル解析
- `ngram_range=(3, 5)`: 3-5文字のn-gram
- `min_df=2`: 最低2回出現
- `max_df=0.95`: 95%以下の文書に出現
- `sublinear_tf=True`: TF値の対数変換
- `norm='l2'`: L2正規化

### 🎯 優先度3: 学習器改善
**LogisticRegression設定**:
```python
LogisticRegression(
    class_weight='balanced',  # クラス不均衡対応
    max_iter=200,            # 収束まで十分な反復
    n_jobs=-1,               # 並列処理
    random_state=42          # 再現性確保
)
```

**ハイパーパラメータ探索**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__C': [0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5))
```

### 🎯 優先度4: 確率校正
**CalibratedClassifierCV導入**:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(
    grid_search.best_estimator_,
    method='sigmoid',
    cv=3
)
```

### 🎯 優先度5: 閾値最適化
**PR曲線使用**:
```python
from sklearn.metrics import precision_recall_curve

# F1最大化閾値
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

## 実装ファイル修正計画

### 1. model_training.py の全面改修
**現在の問題**:
- 全データでTF-IDFをfit
- 単純なtrain_test_split使用
- ハイパーパラメータ固定

**改善後**:
- Pipeline + StratifiedKFold
- GridSearchCV + 確率校正
- 閾値最適化

### 2. model_loader.py の更新
**追加機能**:
- 校正済みモデルの読み込み
- 最適閾値の適用
- 確率スコアの改善

### 3. テストの更新
**test_model_accuracy.py**:
- 新しいモデルでの検証
- 改善結果の確認

## 技術仕様

### 新しい依存関係
```toml
# pyproject.toml に追加不要（既存のscikit-learnに含まれる）
# - Pipeline
# - StratifiedKFold  
# - GridSearchCV
# - CalibratedClassifierCV
```

### モデル保存形式
```python
# 校正済みモデル + 最適閾値
model_data = {
    'calibrated_model': calibrated_clf,
    'optimal_threshold': optimal_threshold,
    'feature_names': feature_names,
    'label_encoder': label_encoder
}
```

## 検証方法

### 実行コマンド
```bash
# 新しいモデル訓練
cd backend
poetry run python scripts/model_training.py

# Phase 1テストで検証
poetry run pytest tests/test_model_accuracy.py -v

# メモリ・予測テスト
poetry run python test_memory_and_prediction.py
```

### 成功基準
1. **サニティテスト**: 2つのテキストが異なるラベルで分類
2. **Baseline比較**: macro F1 > 0.383（DummyClassifier超え）
3. **混同行列**: 両クラスに適切に分布
4. **スコア分布**: 0.3-0.7の範囲に広がり

### 期待される改善
- **Accuracy**: 0.379 → 0.60+ (目標)
- **Macro F1**: 0.276 → 0.50+ (目標)
- **ポジティブ正解率**: 0.1% → 70%+ (目標)
- **スコア範囲**: 0.038 → 0.4+ (目標)

## 実装順序

### Step 1: Pipeline実装
1. model_training.py の Pipeline化
2. StratifiedKFold導入
3. データリーク防止確認

### Step 2: ハイパーパラメータ探索
1. GridSearchCV実装
2. TF-IDFパラメータ最適化
3. LogisticRegression C値調整

### Step 3: 確率校正
1. CalibratedClassifierCV導入
2. 確率分布の改善確認
3. スコア範囲の拡大

### Step 4: 閾値最適化
1. PR曲線での最適閾値決定
2. F1スコア最大化
3. 予測精度の向上

### Step 5: 検証とテスト
1. Phase 1テストでの検証
2. サニティテストの通過確認
3. 全体的な性能改善確認

## リスク管理

### 技術的リスク
- **訓練時間増加**: GridSearchCV + 確率校正で時間延長
- **メモリ使用量**: Pipeline化でメモリ増加の可能性
- **互換性**: 既存のmodel_loaderとの互換性確保

### 対策
- **段階的実装**: 各ステップごとに検証
- **バックアップ**: 既存モデルの保持
- **テスト駆動**: 各改善後にテスト実行

## 成果物予定

### 修正ファイル
- `backend/scripts/model_training.py`: 全面改修
- `backend/app/model_loader.py`: 校正モデル対応
- `backend/tests/test_model_accuracy.py`: 新モデル検証

### 新規ファイル
- `docs/model-improvement-phases/phase2_completion_report.md`: 完了報告

### 分析結果
- 改善後の混同行列・スコア分布
- Before/After比較レポート
- 特徴語分析の更新

---

**作成日**: 2025年8月25日  
**作成者**: Devin AI  
**プロジェクト**: 日本語感情分析モデル改善  
**フェーズ**: Phase 2 - モデル改善実装
