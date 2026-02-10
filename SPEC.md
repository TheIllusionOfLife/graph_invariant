# 実装仕様書: LLM駆動型グラフ不変量自動発見システム

> **文書の位置づけ:** 本文書は、初期提案書 [Research_Plan_Graph_Invariant_Discovery.md](./Research_Plan_Graph_Invariant_Discovery.md) とそのレビュー [REVIEW.md](./REVIEW.md) を踏まえ、研究者へのインタビューを経て策定した**確定実装仕様**である。レビューで指摘された全課題（セキュリティ、実験設計、評価指標等）に対する設計判断を含む。実装はこの文書に準拠すること。

## 1. 設計判断サマリ

| 項目 | 決定 |
|------|------|
| 相関の方向 | 符号付きで保持。`abs()` は取らず正負を記録。ランキングは絶対値で行う |
| サンドボックス | `subprocess` + `resource` モジュールでプロセスレベル隔離 |
| ベースライン | PySRのみ（シンボリック回帰との直接比較） |
| LLMモデル | ローカル8B (Ollama) をメイン。必要になった場合のみ Gemini API ($10/month上限) |
| 探索戦略 | 完全自由生成（現状維持）。関数シグネチャのみ固定 |
| 淘汰戦略 | 島モデル（Island Model） |
| グラフ規模 | 段階的拡大: Phase 1は N=30〜100、Phase 2で N=30〜1000 |
| 簡潔性スコア | ASTノード数 + SymPy簡約後の文字数を重み付け平均 |
| Phase 1 成功基準 | Valセットでスピアマン相関 ≥ 0.85 |
| 再現性 | 全ログ保存（LLM入出力・シード・評価結果をJSON） |
| OODデータセット | Phase 1の結果を見て後で決定 |
| 成果物 | 論文（arXiv投稿）+ OSSツール公開 |
| コード構成 | モジュール分割 |

---

## 2. 島モデル設計

### 2.1 島構成（4島）

| 島 | Temperature | プロンプト戦略 | 目的 |
|----|-------------|---------------|------|
| Island 0 | 0.3 | 改良特化（既存式のrefine指示） | 局所最適の追求 |
| Island 1 | 0.3 | 組合せ特化（2式を組み合わせる指示） | 既存知識の統合 |
| Island 2 | 0.8 | 改良特化 | バランス型探索 |
| Island 3 | 1.2 | 新規発想（全く新しいアプローチ指示） | 大胆な探索 |

### 2.2 移住（Migration）

- **頻度**: 10世代ごと
- **方法**: 各島のトップ1候補を全他島に共有（リング型トポロジ）
- **移住時の処理**: 受け入れ島の最下位候補と置換（スコア比較後）

### 2.3 淘汰

- 各島で独立にknowledge_baseを管理
- 各島で上位5〜10個を保持（島ごとの多様性は4島の差別化で確保）

---

## 3. データ分割

### Phase 1 (N=30〜100)

| セット | サイズ | 用途 |
|--------|--------|------|
| Train | 50 | 高速スクリーニング |
| Validation | 200 | 世代選抜（知識ベース更新の判断） |
| Test | 200 | 最終評価専用（選抜に一切使わない） |

### Phase 2 (N=30〜1000)

- 発見された式のスケーリング挙動検証用に別途生成
- Test セットを拡張（N=100〜1000の大規模グラフを追加）

---

## 4. 評価指標

### 4.1 精度

- **主指標**: スピアマン順位相関係数 $\rho$ （符号付き保持、ランキングは $|\rho|$）
- **補助指標**: ピアソン相関係数 $r$、RMSE、MAE

### 4.2 簡潔性スコア

```
simplicity_score = w1 * (1 / ast_node_count) + w2 * (1 / sympy_simplified_length)
```

- `ast_node_count`: Python ASTのノード数（コード複雑度）
- `sympy_simplified_length`: SymPyで簡約化後の数式文字列長（数学的簡潔性）
- 重み `w1=0.5, w2=0.5` を初期値とし、チューニング可能に設計

### 4.3 新規性

- 発見された式と既存不変量（density, clustering_coefficient, degree_assortativity, transitivity）との相関行列を算出
- 全既存指標との $|\rho| < 0.7$ を「新規」と判定

### 4.4 総合スコア

```
total_score = alpha * |spearman_corr| + beta * simplicity_score + gamma * novelty_bonus
```

- 初期値: `alpha=0.6, beta=0.2, gamma=0.2`

---

## 5. サンドボックス設計

### 5.1 プロセス隔離

```
subprocess.run() で各コード実行を別プロセスに分離
├── __builtins__ = {} でビルトイン遮断
├── ホワイトリスト: np, nx, math, abs, min, max, sum, len, sorted, range, enumerate
├── resource.setrlimit() で制約:
│   ├── CPU時間: TIMEOUT_SEC (2秒)
│   └── メモリ: 256MB
└── エラー時は None を返す（0.0 ではなく）
```

### 5.2 禁止パターン

- `import`, `__import__`, `eval`, `exec`, `open`, `os`, `sys`, `subprocess` を文字列レベルで事前検出・拒否

---

## 6. 試行規模

### Phase 1

- **世代数**: 最大20世代（早期停止付き）
- **早期停止条件**: 全島で10世代連続スコア改善なし
- **個体数**: 5個体/島/世代
- **初回総試行**: 4島 × 20世代 × 5個体 = 最大400回のLLM呼び出し
- スコア改善が見られれば段階的に世代数をスケール

### Phase 2

- Phase 1の結果を踏まえてスケール決定
- 最大100世代 × 5個体/島 を上限目安

---

## 7. ログ設計

### 保存形式: JSON Lines (.jsonl)

各LLM呼び出しごとに1レコード:

```json
{
  "generation": 5,
  "island": 2,
  "timestamp": "2025-01-15T10:30:00",
  "prompt": "...",
  "llm_response": "...",
  "extracted_code": "...",
  "train_score": 0.72,
  "val_score": 0.68,
  "simplicity": {"ast_nodes": 15, "sympy_length": 23},
  "status": "success",
  "error": null
}
```

### グラフ生成シード

- データセット生成時のマスターシードを記録
- 各グラフの個別シードもログに含める

---

## 8. モジュール構成

```
graph_invariant/
├── REVIEW.md
├── SPEC.md
├── Research_Plan_Graph_Invariant_Discovery.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config.py          # 全設定値（データサイズ、世代数、温度等）
│   ├── graph_generator.py # グラフ生成・データセット管理
│   ├── generator.py       # LLM呼び出し・コード生成
│   ├── sandbox.py         # subprocess隔離実行・安全性チェック
│   ├── evaluator.py       # 仮説評価（相関計算・簡潔性スコア）
│   ├── knowledge_base.py  # 島モデル・淘汰・移住ロジック
│   ├── logger.py          # JSONL ログ管理
│   └── main.py            # メインループ（進化ループ制御）
├── baselines/
│   └── pysr_baseline.py   # PySRベースライン比較
└── tests/
    ├── test_graph_generator.py
    ├── test_sandbox.py
    ├── test_evaluator.py
    └── test_knowledge_base.py
```

---

## 9. Phase 1 ターゲットと成功基準

- **ターゲット**: `average_shortest_path_length`
- **成功基準**: Validation セットでスピアマン相関 ≥ 0.85（かつ PySR と同等以上）
- **Phase 2 への移行条件**: 上記を満たし、Test セットでも ≥ 0.80

---

## 10. 未決定事項（Phase 1完了後に決定）

- OOD検証用の実世界データセット選定
- Gemini API の導入タイミングと利用戦略
- Phase 2 のターゲット（algebraic_connectivity で確定か、他の候補も検討するか）
- 論文の投稿先（Workshop or Main conference）
