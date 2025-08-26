# Phase 3 計画: 2クラス分類の高度な最適化と本格運用準備

## 概要
**Repository**: ympnov22/japanese-sentiment-analyzer  
**Branch**: devin/1756116053-model-accuracy-improvement  
**Phase**: Phase 3 - 2クラス分類の高度な最適化と本格運用準備  
**計画作成日**: 2025年8月25日  
**計画修正日**: 2025年8月25日（ユーザー承認済み）

## 🎯 Phase 3 の目標

Phase 2でバイアス問題を解決し基本性能を改善したため、Phase 3では既存の2クラス分類データセットを活用した高度な最適化を実施します：

### 主要目標
1. **アンサンブル手法の導入** - 複数モデルの組み合わせで精度向上
2. **特徴エンジニアリング強化** - 日本語特有の特徴量追加
3. **本格運用準備** - パフォーマンス最適化とモニタリング
4. **API拡張** - バッチ処理とリアルタイム分析機能
5. **推論最適化** - 速度・メモリ効率の向上

## 📋 Phase 3 タスクリスト

### Task 3.1: アンサンブル手法導入
- [ ] 複数アルゴリズムの実装（SVM, Random Forest, XGBoost）
- [ ] Voting Classifier の実装
- [ ] Stacking Ensemble の実装
- [ ] アンサンブル性能評価とベンチマーク
- [ ] 最適なアンサンブル構成の決定

### Task 3.2: 特徴エンジニアリング強化
- [ ] 感情語辞書の統合（日本語感情極性辞書）
- [ ] 品詞情報の活用（MeCab統合）
- [ ] 文長・句読点などの統計的特徴量
- [ ] Word2Vec/FastTextによる分散表現
- [ ] 特徴量重要度分析と選択

### Task 3.3: パフォーマンス最適化
- [ ] モデル推論速度の最適化
- [ ] メモリ使用量の削減
- [ ] バッチ処理機能の実装
- [ ] キャッシュ機能の追加
- [ ] 並列処理対応

### Task 3.4: モニタリング・ロギング強化
- [ ] 予測信頼度の詳細ログ
- [ ] モデル性能メトリクスの継続監視
- [ ] データドリフト検出機能
- [ ] A/Bテスト基盤の準備
- [ ] アラート機能の実装

### Task 3.5: API機能拡張
- [ ] バッチ分析エンドポイント（複数テキスト一括処理）
- [ ] ストリーミング分析対応
- [ ] 分析結果の詳細出力（特徴量寄与度など）
- [ ] カスタム閾値設定機能
- [ ] 分析履歴の保存・検索機能

## 🔧 技術仕様

### 3.1 アンサンブル構成
```python
# Voting Ensemble
ensemble = VotingClassifier([
    ('lr', LogisticRegression(...)),
    ('svm', SVC(probability=True, ...)),
    ('rf', RandomForestClassifier(...)),
    ('xgb', XGBClassifier(...))
], voting='soft')

# Stacking Ensemble
stacking = StackingClassifier([
    ('lr', LogisticRegression(...)),
    ('svm', SVC(probability=True, ...)),
    ('rf', RandomForestClassifier(...))
], final_estimator=LogisticRegression())
```

### 3.2 特徴エンジニアリング
```python
# 複合特徴量パイプライン
feature_union = FeatureUnion([
    ('tfidf_char', TfidfVectorizer(analyzer='char', ...)),
    ('tfidf_word', TfidfVectorizer(analyzer='word', ...)),
    ('sentiment_dict', SentimentDictionaryTransformer()),
    ('pos_features', POSFeatureTransformer()),
    ('statistical', StatisticalFeatureTransformer())
])
```

### 3.3 API拡張仕様
```python
# バッチ分析エンドポイント
@app.post("/analyze/batch")
async def analyze_batch(texts: List[str]) -> List[SentimentResult]:
    pass

# 詳細分析エンドポイント
@app.post("/analyze/detailed")
async def analyze_detailed(text: str) -> DetailedSentimentResult:
    # 特徴量寄与度、信頼度詳細、代替予測などを含む
    pass
```

## 📊 評価基準

### 3.1 アンサンブル性能目標
- **単一モデル比較**: +5%以上の性能向上
- **推論時間**: 単一モデルの2倍以内
- **メモリ使用量**: 単一モデルの3倍以内

### 3.2 パフォーマンス目標
- **推論速度**: < 100ms/request（単一テキスト）
- **バッチ処理**: > 1000 texts/second
- **メモリ使用量**: < 500MB（本格運用時）
- **CPU使用率**: < 70%（通常負荷時）

## 🧪 検証戦略

### 3.1 アンサンブル検証
```python
# アンサンブル vs 単一モデル比較
ensemble_score = evaluate_model(ensemble_model, test_data)
single_score = evaluate_model(best_single_model, test_data)
assert ensemble_score > single_score + 0.05  # 5%以上向上
```

### 3.2 パフォーマンス検証
```python
# 推論速度テスト
start_time = time.time()
result = model.predict(test_text)
inference_time = time.time() - start_time
assert inference_time < 0.1  # 100ms以内

# バッチ処理テスト
start_time = time.time()
results = model.predict_batch(test_texts_1000)
batch_time = time.time() - start_time
throughput = 1000 / batch_time
assert throughput > 1000  # 1000 texts/second以上
```

## 📁 成果物

### コードファイル
- `backend/scripts/ensemble_training.py`: アンサンブル訓練スクリプト
- `backend/scripts/feature_engineering.py`: 特徴エンジニアリング
- `backend/app/models/`: 新しいモデルクラス群
- `backend/app/services/`: 拡張されたサービス層

### テストファイル
- `backend/tests/test_ensemble.py`: アンサンブルテスト
- `backend/tests/test_performance.py`: パフォーマンステスト
- `backend/tests/test_api_extended.py`: 拡張API テスト

### ドキュメント
- `docs/model-improvement-phases/phase3_completion_report.md`: 完了報告
- `docs/api/`: API仕様書更新
- `docs/performance/`: パフォーマンス分析結果

## 🔄 実装順序

### Week 1: アンサンブル手法
1. 複数アルゴリズムの個別実装
2. Voting Classifier実装
3. Stacking Ensemble実装
4. アンサンブル性能評価

### Week 2: 特徴エンジニアリング
1. 感情語辞書統合
2. 品詞情報活用（MeCab）
3. 統計的特徴量追加
4. 分散表現統合

### Week 3: 最適化・運用準備
1. パフォーマンス最適化
2. API機能拡張
3. モニタリング機能実装
4. 総合テスト・ベンチマーク

## ⚠️ リスク・課題

### 技術的リスク
- **計算量増加**: アンサンブルによる推論時間増加
- **メモリ使用量**: 複数モデル保持によるメモリ圧迫
- **複雑性**: システム全体の複雑性増加

### 対策
- **段階的実装**: 機能を段階的に追加し、各段階で性能評価
- **プロファイリング**: 詳細な性能分析とボトルネック特定
- **フォールバック**: 単一モデルへのフォールバック機能

## 🎯 成功基準

Phase 3完了時に以下をすべて達成：

### 機能面
- ✅ 3クラス分類が正確に動作
- ✅ アンサンブルが単一モデルを上回る性能
- ✅ 拡張APIが正常に動作
- ✅ バッチ処理機能が動作

### 性能面
- ✅ Macro F1 > 0.75（3クラス）
- ✅ 推論速度 < 100ms
- ✅ バッチ処理 > 1000 texts/second
- ✅ メモリ使用量 < 500MB

### 運用面
- ✅ モニタリング機能が動作
- ✅ ログ機能が充実
- ✅ エラーハンドリングが適切
- ✅ ドキュメントが完備

## 📞 承認プロセス

Phase 3実装前に以下の承認を取得：
1. **技術仕様の承認**: アーキテクチャ・実装方針
2. **性能目標の承認**: 具体的な数値目標
3. **実装スケジュールの承認**: 段階的実装計画

---

**作成日**: 2025年8月25日  
**作成者**: Devin AI  
**プロジェクト**: 日本語感情分析モデル改善  
**フェーズ**: Phase 3 - 高度なモデル最適化と本格運用準備

**Link to Devin run**: https://app.devin.ai/sessions/5c2503a4e73c472dbd21f752507963b6  
**Requested by**: ヤマシタ　ヤスヒロ (@ympnov22)
