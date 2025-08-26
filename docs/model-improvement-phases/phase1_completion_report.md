# Phase 1 完了報告: テスト強化とバイアス分析

## 実行日時
2025年8月25日 10:05 UTC

## 完了したタスク

### ✅ 1. サニティテスト追加
- **実装**: `backend/tests/test_model_accuracy.py` の `test_sanity_check()` メソッド
- **テスト内容**: 
  - ポジティブテキスト: "最高に嬉しい！"
  - ネガティブテキスト: "最悪で腹が立つ。"
- **結果**: **🚨 BIAS DETECTED** - 両方とも「ネガティブ」と分類（スコア: 0.500）

### ✅ 2. 新しい精度テスト実装
- **ファイル**: `backend/tests/test_model_accuracy.py` 作成完了
- **評価指標**: accuracy, precision, recall, macro F1
- **混同行列**: 生成・可視化完了 (`outputs/confusion_matrix.png`)
- **確率分布ヒストグラム**: 生成完了 (`outputs/score_distribution.png`)
- **ベースライン比較**: DummyClassifier(strategy="most_frequent") との比較実装

### ✅ 3. エラー分析実装
- **誤分類Top20件**: 出力完了 (`outputs/error_analysis.json`)
- **特徴語分析**: coef_ を用いた上位/下位係数分析実装
- **エラー率**: 62.1% (910/1465件の誤分類)

## 🚨 重大な問題発見

### バイアス検出結果
```
Sanity Test Results:
Positive text: '最高に嬉しい！' -> ネガティブ (score: 0.500)
Negative text: '最悪で腹が立つ。' -> ネガティブ (score: 0.500)
BIAS DETECTED: Both texts classified as 'ネガティブ' - Model shows clear bias!
```

### 性能指標（現在のモデル vs ベースライン）
| 指標 | 現在のモデル | DummyClassifier | 差分 |
|------|-------------|----------------|------|
| **Accuracy** | 0.379 | 0.620 | **-0.241** |
| **Macro F1** | 0.276 | 0.383 | **-0.107** |
| **Precision (macro)** | 0.315 | - | - |
| **Recall (macro)** | 0.498 | - | - |

### 混同行列
```
Labels: ['ネガティブ', 'ポジティブ']
[[554   3]
 [907   1]]
```
- **ネガティブ**: 557件中554件正解（99.5%）
- **ポジティブ**: 908件中1件正解（0.1%）← **極めて深刻**

### スコア分布分析
- **範囲**: 0.500 - 0.538（極めて狭い）
- **平均**: 0.500
- **標準偏差**: 0.002
- **⚠️ 警告**: スコア範囲が0.038と非常に狭く、明確なバイアスを示している

## 詳細分析結果

### 1. メモリ・予測テスト結果
```
=== Memory and Prediction Analysis ===
1. Text: この商品は本当に素晴らしいです！最高の品質で大満足です。
   Result: ポジティブ (confidence: 0.736) ← 正常
2. Text: 普通の商品だと思います。特に良くも悪くもありません。
   Result: ネガティブ (confidence: 0.500) ← バイアス
3. Text: 最悪の商品でした。二度と買いません。お金の無駄でした。
   Result: ネガティブ (confidence: 0.749) ← 正常
4. Text: まあまあです。
   Result: ネガティブ (confidence: 0.500) ← バイアス
5. Text: とても良い！
   Result: ネガティブ (confidence: 0.500) ← バイアス
6. Text: ひどい。
   Result: ネガティブ (confidence: 0.500) ← バイアス
```

### 2. 誤分類Top5例
1. **True: ネガティブ | Pred: ポジティブ | Conf: 0.538**
   - Text: "変える手間を気にしないのであれば、普通のを二つ買ったほうがよさげ。"
2. **True: ネガティブ | Pred: ポジティブ | Conf: 0.529**
   - Text: "プリンタが安価なのにインクが高い。。。"
3. **True: ポジティブ | Pred: ネガティブ | Conf: 0.523**
   - Text: "簡単に貼れて良かったです。"
4. **True: ネガティブ | Pred: ポジティブ | Conf: 0.510**
   - Text: "りけいシート１も２も非常に剥がし辛い"
5. **True: ポジティブ | Pred: ネガティブ | Conf: 0.500**
   - Text: "携帯ショップでブラックの本体しか残っていなかったので仕方なく購入。"

### 3. 特徴語分析
- **Top positive coefficients**: 最大値 1.057（20個の特徴語特定済み）
- **Bottom negative coefficients**: 最小値 -0.997（20個の特徴語特定済み）
- **モデル係数**: 利用可能、Phase 2での詳細分析に活用可能

## 問題の根本原因分析

### 1. データリークの可能性
- 現在の実装では全データでTF-IDFをfitしてから分割している可能性
- 検証・テストデータに情報が漏れている

### 2. クラス不均衡問題
- テストデータ: ネガティブ557件 vs ポジティブ908件
- `class_weight="balanced"` が適切に機能していない

### 3. TF-IDFパラメータの問題
- 現在の設定が日本語テキストに最適化されていない
- n-gramの範囲、min_df、max_dfの調整が必要

### 4. 確率校正の欠如
- 予測確率が0.5付近に集中
- CalibratedClassifierCVが未実装

## Phase 2への提案

### 🎯 優先度1: パイプライン化とリーク防止
1. **Pipeline + StratifiedKFold実装**
   - 各fold内でfit/transformを実行
   - データリークを完全に防止

### 🎯 優先度2: TF-IDF最適化
1. **推奨初期設定**:
   ```python
   TfidfVectorizer(
       analyzer='char',
       ngram_range=(3, 5),
       min_df=2,
       max_df=0.95,
       sublinear_tf=True,
       norm='l2'
   )
   ```

### 🎯 優先度3: 学習器改善
1. **LogisticRegression設定**:
   ```python
   LogisticRegression(
       class_weight='balanced',
       max_iter=200,
       n_jobs=-1,
       random_state=42
   )
   ```
2. **ハイパーパラメータ探索**: Cパラメータのチューニング
3. **CalibratedClassifierCV導入**: 確率校正

### 🎯 優先度4: 閾値最適化
1. **PR曲線使用**: F1最大化またはYouden's J
2. **検証セットでの最適閾値決定**

## 成功基準の現状

| 基準 | 現状 | 目標 | 状態 |
|------|------|------|------|
| サニティテスト区別 | ❌ 両方ネガティブ | ✅ 正しく区別 | **要改善** |
| macro F1 > baseline | ❌ 0.276 < 0.383 | ✅ > 0.383 | **要改善** |
| 全件ポジティブ解消 | ❌ 全件ネガティブ | ✅ 両クラス分布 | **要改善** |
| 確率分布の広がり | ❌ 0.500-0.538 | ✅ 0.3-0.7以外 | **要改善** |

## 生成ファイル

### テストファイル
- ✅ `backend/tests/test_model_accuracy.py` - 包括的な精度テストスイート

### 出力ファイル
- ✅ `backend/outputs/confusion_matrix.png` - 混同行列可視化
- ✅ `backend/outputs/score_distribution.png` - スコア分布ヒストグラム
- ✅ `backend/outputs/error_analysis.json` - 詳細エラー分析

## 検証コマンド実行結果

### ✅ pytest実行
```bash
poetry run pytest tests/test_model_accuracy.py -v
# 全テスト実行完了、バイアス検出成功
```

### ✅ メモリテスト実行
```bash
poetry run python test_memory_and_prediction.py
# メモリ使用量: +3.7MB、予測機能正常動作確認
```

## 次のステップ

**🔄 Phase 2承認待ち**

Phase 1で特定された重大なバイアス問題と性能劣化を解決するため、Phase 2のモデル改善実装の承認をお願いします。

**Phase 2実装予定**:
1. Pipeline + StratifiedKFold実装（データリーク防止）
2. TF-IDFパラメータ最適化
3. LogisticRegression + ハイパーパラメータ探索
4. CalibratedClassifierCV導入
5. 閾値最適化

---

**Repository**: ympnov22/japanese-sentiment-analyzer  
**Branch**: devin/1756116053-model-accuracy-improvement  
**Commit Hash**: [次のcommit後に更新]  
**Link to Devin run**: https://app.devin.ai/sessions/5c2503a4e73c472dbd21f752507963b6  
**Requested by**: ヤマシタ ヤスヒロ (@ympnov22)
