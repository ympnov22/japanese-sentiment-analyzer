# 日本語感情分析モデル - 詳細テスト計画
# Comprehensive Testing Plan for Japanese Sentiment Analysis Model

## 1. テスト目標 (Testing Objectives)

### 主要目標 (Primary Goals)
- バイナリ分類精度の検証と改善（目標: >95%）
- クロスバリデーションによる汎化性能の評価
- メモリ使用量の継続的監視（目標: <400MB）
- 本番環境での安定性確認

### 評価指標 (Evaluation Metrics)
- **Accuracy**: 全体的な分類精度
- **Precision**: 精密度（偽陽性の最小化）
- **Recall**: 再現率（偽陰性の最小化）
- **F1 Score**: 精密度と再現率の調和平均
- **ROC AUC**: 受信者操作特性曲線下面積
- **Confusion Matrix**: 混同行列による詳細分析

## 2. テストデータセット (Test Datasets)

### データソース (Data Sources)
1. **Hugging Face Dataset**: `sepidmnorozy/Japanese_sentiment`
   - サイズ: ~10,000サンプル
   - 品質: 高品質な人手ラベル
   - バランス: ポジティブ/ネガティブ均等

2. **拡張合成データ**: Enhanced Synthetic Data
   - 多様な日本語表現パターン
   - 短文から長文まで幅広い長さ
   - 実際の商品レビュー風テキスト

### データ前処理 (Data Preprocessing)
- 日本語テキスト正規化
- 重複句読点の統一
- 空白文字の正規化
- 長さフィルタリング（5-500文字）

## 3. クロスバリデーション戦略 (Cross-Validation Strategy)

### 5-Fold Stratified Cross-Validation
- **分割数**: 5フォールド
- **層化**: クラス比率を維持
- **シャッフル**: ランダムシード固定（再現性確保）
- **評価**: 各フォールドで全指標を計算

### バリデーション指標
```python
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted', 
    'f1': 'f1_weighted',
    'roc_auc': 'roc_auc'
}
```

## 4. ハイパーパラメータ最適化 (Hyperparameter Optimization)

### グリッドサーチ範囲
```python
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],           # 正則化強度
    'max_iter': [1000, 2000, 5000],                # 最大反復数
    'solver': ['liblinear', 'lbfgs'],              # ソルバー
    'penalty': ['l1', 'l2']                        # 正則化タイプ
}
```

### 最適化戦略
- **目標指標**: F1スコア（weighted）
- **検証方法**: 5-fold CV
- **並列処理**: 全CPUコア使用
- **制約**: l1正則化はliblinearソルバーのみ

## 5. 性能分析 (Performance Analysis)

### エラー分析 (Error Analysis)
- 誤分類サンプルの詳細調査
- 信頼度スコア分布の分析
- 困難なケースの特定
- テキスト長と精度の関係

### 統計的有意性検定
- McNemar検定による改善の有意性
- 信頼区間の計算
- 複数モデル間の比較

## 6. メモリ効率テスト (Memory Efficiency Testing)

### メモリ監視項目
- 初期メモリ使用量
- モデル読み込み時のメモリスパイク
- 予測実行時のメモリ使用量
- ガベージコレクション効果

### 目標値
- **ピーク使用量**: <400MB
- **常時使用量**: <200MB
- **読み込み増加**: <50MB

## 7. 本番環境テスト (Production Environment Testing)

### Fly.io デプロイテスト
- OOMエラーの発生確認
- API応答時間の測定
- 同時リクエスト処理能力
- ヘルスチェック安定性

### 負荷テスト
- 連続予測リクエスト
- メモリリーク検出
- CPU使用率監視
- レスポンス時間分布

## 8. 回帰テスト (Regression Testing)

### 既存機能の確認
- API エンドポイント動作
- レスポンス形式の一貫性
- エラーハンドリング
- ログ出力の正常性

### 比較ベースライン
- 元のモデル（90.4%精度）との比較
- メモリ使用量の改善確認
- 予測速度の維持/改善

## 9. テスト実行スケジュール (Test Execution Schedule)

### Phase 1: データ準備とクロスバリデーション (30分)
1. データセット読み込み
2. 前処理実行
3. 5-fold CV実行
4. 基本性能指標計算

### Phase 2: ハイパーパラメータ最適化 (45分)
1. グリッドサーチ実行
2. 最適パラメータ特定
3. 最適モデル訓練
4. 性能比較

### Phase 3: 詳細分析 (30分)
1. エラー分析実行
2. 統計的検定
3. メモリ効率測定
4. 結果レポート生成

### Phase 4: 本番環境検証 (15分)
1. Fly.io デプロイ
2. API動作確認
3. 負荷テスト実行
4. 最終検証

## 10. 成功基準 (Success Criteria)

### 必須要件 (Required)
- ✅ クロスバリデーション F1 > 0.95
- ✅ テスト精度 > 95%
- ✅ メモリ使用量 < 400MB
- ✅ Fly.io 安定デプロイ

### 推奨要件 (Preferred)
- 🎯 ROC AUC > 0.98
- 🎯 エラー率 < 3%
- 🎯 API応答時間 < 2秒
- 🎯 メモリ使用量 < 200MB

## 11. レポート形式 (Report Format)

### 自動生成レポート
```json
{
  "test_date": "2025-08-24T22:00:00",
  "cross_validation": {
    "folds": 5,
    "metrics": {
      "accuracy": {"mean": 0.xxx, "std": 0.xxx},
      "f1": {"mean": 0.xxx, "std": 0.xxx}
    }
  },
  "best_hyperparameters": {...},
  "final_performance": {...},
  "memory_usage": {...}
}
```

### 可視化出力
- 混同行列ヒートマップ
- ROC曲線
- 精度-再現率曲線
- メモリ使用量推移グラフ

この包括的テスト計画により、モデルの性能と安定性を多角的に評価し、本番環境での信頼性を確保します。
