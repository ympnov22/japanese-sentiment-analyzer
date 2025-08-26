# Phase 1 実装計画: テスト強化とバイアス分析

## 概要
日本語感情分析モデルのバイアス修正と精度改善プロジェクトのPhase 1実装計画

**対象リポジトリ**: ympnov22/japanese-sentiment-analyzer  
**ブランチ**: devin/1756116053-model-accuracy-improvement  
**実装期間**: 2025年8月25日

## 背景と問題認識

### 現在の問題
- すべての入力が「ポジティブ」と判定される（ユーザー報告）
- スコアが0.38〜0.59に固定される
- 実用的な分類器として機能していない

### 解決すべき課題
1. **バイアス検出**: 固定テストでバイアスを即座に検出
2. **性能評価**: 現在のモデルの実際の性能を定量化
3. **ベースライン比較**: ランダム分類器との比較
4. **エラー分析**: 誤分類パターンの特定

## Phase 1 実装タスク

### 1. サニティテスト追加
**目的**: バイアス検出の自動化

**実装内容**:
```python
def test_sanity_check():
    positive_text = "最高に嬉しい！"
    negative_text = "最悪で腹が立つ。"
    
    # 両方が同じラベルならバイアス検出
    if positive_result['result'] == negative_result['result']:
        pytest.fail("BIAS DETECTED")
```

**成功基準**: 2つのテキストが異なるラベルで分類される

### 2. 新しい精度テスト実装
**ファイル**: `backend/tests/test_model_accuracy.py`

**評価指標**:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)

**可視化機能**:
- 混同行列ヒートマップ
- 確率分布ヒストグラム

**ベースライン比較**:
```python
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
```

### 3. エラー分析機能
**実装機能**:
- 誤分類Top20件の出力
- 信頼度スコア順でのソート
- 特徴語分析（coef_使用）

**出力形式**:
```json
{
  "misclassifications": [
    {
      "true_label": "ポジティブ",
      "predicted_label": "ネガティブ",
      "confidence": 0.523,
      "text": "..."
    }
  ],
  "feature_analysis": {
    "top_positive_features": [...],
    "top_negative_features": [...]
  }
}
```

## 技術仕様

### テスト環境
- **Python**: 3.11+
- **テストフレームワーク**: pytest
- **可視化**: matplotlib, seaborn
- **機械学習**: scikit-learn

### ファイル構成
```
backend/
├── tests/
│   └── test_model_accuracy.py  # 新規作成
├── outputs/                    # 新規作成
│   ├── confusion_matrix.png
│   ├── score_distribution.png
│   └── error_analysis.json
└── phase1_completion_report.md # 新規作成
```

### 依存関係
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.24.0"
matplotlib = "^3.10.5"
seaborn = "^0.13.2"
```

## 実装ルール

### コード品質
- すべての`random_state=42`で固定
- 型ヒント使用
- docstring記述
- エラーハンドリング実装

### テスト設計
- 各テストは独立して実行可能
- モデル読み込み失敗時のスキップ処理
- 詳細な出力とログ

### 出力管理
- 可視化ファイルは`outputs/`ディレクトリに保存
- JSON形式でのデータ出力
- 日本語文字化け対策

## 検証方法

### 実行コマンド
```bash
# Phase 1テスト実行
cd backend
poetry run pytest tests/test_model_accuracy.py -v

# メモリ・予測テスト
poetry run python test_memory_and_prediction.py
```

### 成功基準
1. **サニティテスト**: 2つのテキストが異なるラベルで分類
2. **ベースライン比較**: DummyClassifierとの性能比較完了
3. **可視化**: 混同行列とスコア分布の画像生成
4. **エラー分析**: Top20誤分類例と特徴語リスト出力

### 失敗検出
- 両方のサニティテストが同じラベル → バイアス検出
- モデルF1 < ベースラインF1 → 性能劣化検出
- スコア範囲が狭い（< 0.1） → バイアス警告

## Phase 2への準備

### 特定すべき問題
1. **データリーク**: 全データでTF-IDF fit後の分割
2. **クラス不均衡**: class_weight設定の効果
3. **パラメータ**: TF-IDFとLogisticRegressionの設定
4. **確率校正**: 予測確率の偏り

### 収集すべきデータ
- 現在のモデル係数（coef_）
- 特徴語の重要度ランキング
- クラス別の予測分布
- エラーパターンの分析

## リスク管理

### 技術的リスク
- **モデル読み込み失敗**: スキップ処理で対応
- **メモリ不足**: バッチ処理での予測実行
- **日本語表示問題**: フォント設定とエンコーディング対応

### スケジュールリスク
- **テストデータ不足**: 既存データでの実行
- **環境問題**: poetry環境での実行確認

## 成果物

### コードファイル
- `backend/tests/test_model_accuracy.py`: 包括的テストスイート

### 分析結果
- `backend/outputs/confusion_matrix.png`: 混同行列可視化
- `backend/outputs/score_distribution.png`: スコア分布
- `backend/outputs/error_analysis.json`: 詳細エラー分析

### ドキュメント
- `backend/phase1_completion_report.md`: 完了報告書

---

**作成日**: 2025年8月25日  
**作成者**: Devin AI  
**プロジェクト**: 日本語感情分析モデル改善  
**フェーズ**: Phase 1 - テスト強化とバイアス分析
