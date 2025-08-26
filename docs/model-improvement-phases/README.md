# モデル改善プロジェクト ドキュメント

## 概要
日本語感情分析モデルのバイアス修正と精度改善プロジェクトの全ドキュメントを管理するフォルダです。

## プロジェクト情報
- **対象リポジトリ**: ympnov22/japanese-sentiment-analyzer
- **開始日**: 2025年8月25日
- **担当**: Devin AI
- **依頼者**: ヤマシタ ヤスヒロ (@ympnov22)

## 現在の問題
- すべての入力が「ポジティブ」と判定される
- スコアが0.38〜0.59に固定される
- 実用的な分類器として機能していない

## フェーズ構成

### Phase 1: テスト強化とバイアス分析 ✅
**状態**: 完了  
**期間**: 2025年8月25日

**主要成果**:
- サニティテスト実装（バイアス検出）
- 包括的精度テストスイート作成
- ベースライン比較（DummyClassifier）
- 混同行列・スコア分布可視化
- エラー分析（誤分類Top20・特徴語分析）

**重要な発見**:
- 🚨 **重大バイアス検出**: 両サニティテストが「ネガティブ」分類
- 📉 **性能劣化**: Model F1 (0.276) < Baseline F1 (0.383)
- 📊 **極端な偏り**: ポジティブクラス正解率 0.1% (1/908件)

### Phase 2: モデル改善 🔄
**状態**: 承認待ち  
**予定**: Phase 1承認後

**実装予定**:
- Pipeline + StratifiedKFold（データリーク防止）
- TF-IDF最適化（char n-gram (3,5), 最適パラメータ）
- LogisticRegression + ハイパーパラメータ探索
- CalibratedClassifierCV導入（確率校正）
- 閾値最適化（PR曲線使用）

## ドキュメント一覧

### 計画書
- [`phase1_plan.md`](./phase1_plan.md) - Phase 1実装計画書

### 完了報告書
- [`phase1_completion_report.md`](./phase1_completion_report.md) - Phase 1完了報告書

### 技術資料
- 混同行列可視化: `backend/outputs/confusion_matrix.png`
- スコア分布: `backend/outputs/score_distribution.png`
- エラー分析: `backend/outputs/error_analysis.json`

## 成功基準

### Done定義
1. ✅ サニティテストが正しく区別できる
2. ✅ macro F1がbaseline (DummyClassifier)を大きく上回る
3. ✅ 全件ポジティブ予測が解消され、混同行列が両クラスに分布
4. ✅ 確率スコアが0.3〜0.7に偏らず、分布に広がりが出ている

### 現在の状況
- ❌ サニティテスト: 両方「ネガティブ」分類
- ❌ Baseline比較: 0.276 < 0.383（悪化）
- ❌ クラス分布: ポジティブ 0.1%正解率
- ❌ スコア分布: 0.500-0.538（極めて狭い）

## 検証コマンド

```bash
# Phase 1テスト実行
cd backend
poetry run pytest tests/test_model_accuracy.py -v

# メモリ・予測テスト
poetry run python test_memory_and_prediction.py
```

## Git情報
- **ブランチ**: devin/1756116053-model-accuracy-improvement
- **ベースブランチ**: main
- **最新コミット**: fba6e23

## 関連リンク
- **Devin実行セッション**: https://app.devin.ai/sessions/5c2503a4e73c472dbd21f752507963b6
- **GitHubリポジトリ**: https://github.com/ympnov22/japanese-sentiment-analyzer

---

**最終更新**: 2025年8月25日  
**更新者**: Devin AI
