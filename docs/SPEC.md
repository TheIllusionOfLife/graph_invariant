# 実装仕様書: LLM駆動型グラフ不変量自動発見システム

> **文書の位置づけ:** 本文書は、初期提案書 [Research_Plan_Graph_Invariant_Discovery.md](./Research_Plan_Graph_Invariant_Discovery.md) とそのレビュー [REVIEW.md](./REVIEW.md) を踏まえ、研究者へのインタビューを経て策定した**確定実装仕様**である。レビューで指摘された全課題（セキュリティ、実験設計、評価指標等）に対する設計判断を含む。実装はこの文書に準拠すること。（注: これらの文書は `docs/` ディレクトリにまとめて配置されている。）

## 1. 設計判断サマリ

| 項目 | 決定 |
|------|------|
| 相関の方向 | 符号付きで保持。`abs()` は取らず正負を記録。ランキングは絶対値で行う |
| サンドボックス | `multiprocessing.Pool` による事前起動ワーカー + 制限付き `exec` |
| ベースライン | PySR + 軽量統計ベースライン（RandomForest, 線形回帰） |
| LLMモデル | ローカル8B (Ollama) をメイン。必要になった場合のみ Gemini API ($10/month上限) |
| 探索戦略 | 完全自由生成（現状維持）。関数シグネチャのみ固定。停滞時は制約付き生成にフォールバック |
| 淘汰戦略 | 島モデル（Island Model） |
| グラフ規模 | 段階的拡大: Phase 1は N=30〜100、Phase 2で N=30〜1000 |
| 簡潔性スコア | ASTノード数（対数スケール）+ SymPy簡約後の文字数（対数スケール）を重み付け平均 |
| Phase 1 成功基準 | Valセットでスピアマン相関 ≥ 0.85 |
| 再現性 | 全ログ保存（LLM入出力・シード・評価結果をJSON）。世代ごとのチェックポイント |
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
| Island 3 | 1.2 | 新規発想（全く新しいアプローチ指示）。Gemini APIフォールバック対象 | 大胆な探索 |

### 2.2 移住（Migration）

- **頻度**: 10世代ごと（Phase 1では最大20世代のため移住は1〜2回のみ。Phase 2で世代数を増やす際に効果を発揮する設計）
- **トポロジ**: リング型（Island 0→1→2→3→0）。各島は隣接1島にのみ送出する
- **方法**: 各島のトップ1候補を右隣の島に送出（1対1）
- **移住時の処理**: 受け入れ島の最下位候補と置換（移住候補のスコアが上回る場合のみ）

### 2.3 Gemini APIフォールバック

- **条件**: Island 3（高温・新規発想島）が**3世代連続**で有効な候補（Train スコア > 0.3）を1つも生成できなかった場合
- **動作**: 次の1世代のみ、Island 3のLLMバックエンドをローカル8BからGemini APIに切り替え
- **目的**: ローカルモデルの能力限界で探索が停滞した際に、高品質な多様性を注入する
- **コスト管理**: $10/month上限は変更しない。フォールバック発動は1実験あたり最大3回に制限

### 2.4 生成フォールバック（制約付き生成）

自由生成は8Bモデルの能力限界で停滞するリスクが最も高い。以下のフォールバックプロトコルを設ける:

- **発動条件**: いずれかの島で**5世代連続**、有効な候補（Train スコア > 0.3）を1つも生成できなかった場合
- **動作**: 該当島のプロンプトを**制約付き生成**に切り替える
  - **演算子プール**: `+, -, *, /, log, sqrt, **, sum, mean, max, min`
  - **テンプレート骨格**: `def new_invariant(G): n = G.number_of_nodes(); m = G.number_of_edges(); degrees = [d for _, d in G.degree()]; return f(n, m, degrees)`
  - プロンプトに「上記の演算子とテンプレートを使って新しい不変量を構成せよ」と明示指示
- **解除条件**: 制約付き生成で3世代以内にスコア > 0.3の候補が出た場合、自由生成に復帰
- **Gemini APIフォールバックとの関係**: 制約付き生成を先に試行し、それでも停滞する場合にGemini APIフォールバック（§2.3）を発動する（Island 3のみ）

> **設計意図**: FunSearchはPaLM2規模のモデルを使用しており、8B量子化モデルでの数学的関数の自由生成は未実証。制約付き生成は探索空間を限定するが、8Bモデルでも組合せ的な探索が可能になる。

### 2.5 淘汰

- 各島で独立にknowledge_baseを管理
- 各島で上位5〜10個を保持（島ごとの多様性は4島の差別化で確保）

---

## 3. データ分割

### 3.1 グラフ生成器

以下の5種類を均等に生成する（初期提案書に記載のGeometric Graphを含む）:

| 種類 | NetworkX関数 | パラメータ範囲 |
|------|-------------|---------------|
| Erdos-Renyi | `erdos_renyi_graph(N, p)` | p ∈ [0.05, 0.3] |
| Barabasi-Albert | `barabasi_albert_graph(N, m)` | m ∈ [1, 4] |
| Watts-Strogatz | `watts_strogatz_graph(N, k, p)` | k ∈ [4, 8], p ∈ [0.1, 0.5] |
| Random Geometric | `random_geometric_graph(N, r)` | r ∈ [0.1, 0.4] |
| Stochastic Block Model | `stochastic_block_model(sizes, probs)` | 3ブロック |

### 3.2 Sanity Check セット（実世界グラフ）

探索・選抜には使わないが、パイプラインの健全性検証として初期から組み込む:

- `nx.karate_club_graph()` (34ノード, ソーシャルネットワーク)
- `nx.les_miserables_graph()` (77ノード, 共起ネットワーク)
- `nx.florentine_families_graph()` (15ノード, 歴史的ネットワーク)

### 3.3 Phase 1 (N=30〜100)

| セット | サイズ | 用途 |
|--------|--------|------|
| Train | 50 | 高速スクリーニング |
| Validation | 200 | 世代選抜（知識ベース更新の判断） |
| Test | 200 | 最終評価専用（選抜に一切使わない） |

### 3.4 Phase 2 (N=30〜1000)

- 発見された式のスケーリング挙動検証用に別途生成
- Test セットを拡張（N=100〜1000の大規模グラフを追加）

---

## 4. 評価指標

### 4.1 精度

- **主指標**: スピアマン順位相関係数 $\rho$ （符号付き保持、ランキングは $|\rho|$）
- **補助指標**: ピアソン相関係数 $r$、RMSE、MAE

### 4.2 簡潔性スコア

```
simplicity_score = w1 * (1 / (1 + log2(max(ast_node_count, 1)))) + w2 * (1 / (1 + log2(max(sympy_simplified_length, 1))))
```

- `ast_node_count`: Python ASTのノード数（コード複雑度）
- `sympy_simplified_length`: SymPyで簡約化後の数式文字列長（数学的簡潔性）
- 対数スケールを採用: `1/x` は小さな差を過大評価するが、`1/(1+log2(x))` は中程度の複雑さを許容しつつ指数的な複雑性増加を抑制する
- 両項とも `(0, 1]` の範囲に収まるため、重み `w1=0.5, w2=0.5` で均等に寄与する
- 重みはチューニング可能に設計。Phase 1終了時に分布を確認し必要に応じて調整

### 4.3 新規性

- 発見された式と既存不変量との相関行列を算出
- **参照不変量セット（9種）**:
  - `density` — グラフ密度
  - `clustering_coefficient` — 平均クラスタ係数
  - `degree_assortativity` — 次数相関
  - `transitivity` — 推移性
  - `average_degree` — 平均次数（`2m/n`）
  - `max_degree` — 最大次数
  - `spectral_radius` — 隣接行列の最大固有値（`max(eigenvalues(A))`）
  - `diameter` — グラフ直径（連結成分の最長最短距離）
  - `algebraic_connectivity` — ラプラシアンの第2最小固有値（Fiedler値）
- 各既存指標との相関について、ブートストラップ法（1000回リサンプリング）で95%信頼区間を算出
- 全既存指標との相関の95%信頼区間上限が $|\rho| < 0.7$ を満たす場合に「新規」と判定

### 4.4 新規性ボーナス（novelty_bonus）

```
novelty_bonus = max(0, 1 - max_rho_known)
```

- `max_rho_known`: 全既存指標との $|\rho|$ の最大値
- 範囲: `[0, 1]`。既存指標と全く無相関なら1.0、完全に既存指標と一致なら0.0
- 例: 最も相関の高い既存指標との $|\rho|$ が0.4 → `novelty_bonus = 0.6`

### 4.5 総合スコア

```
total_score = alpha * |spearman_corr| + beta * simplicity_score + gamma * novelty_bonus
```

- 全項が `[0, 1]` 範囲のため、重みが直感的に解釈可能
- 初期値: `alpha=0.6, beta=0.2, gamma=0.2`

---

## 5. サンドボックス設計

### 5.1 アーキテクチャ

`multiprocessing.Pool` で事前にワーカープロセスを起動し、各ワーカー内で制限付き `exec` を実行する。`subprocess.run()` をグラフごとに呼ぶ方式は、プロセス起動オーバーヘッド（約50-100ms/回）により1世代あたり数分のボトルネックとなるため採用しない。

```
multiprocessing.Pool(initializer=worker_init)  # ワーカーを事前起動
├── worker_init(): ワーカー内で resource.setrlimit() を設定
│   ├── CPU時間: TIMEOUT_SEC (2秒)
│   └── メモリ: 256MB
├── exec(code, safe_globals) をワーカー内で実行
│   ├── safe_globals = {"__builtins__": {}, ...ホワイトリスト}
│   └── ホワイトリスト: np, nx, math, abs, min, max, sum, len, sorted, range, enumerate
├── signal.alarm(TIMEOUT_SEC) で個別タイムアウト
└── エラー時は None を返す（0.0 ではなく）
```

### 5.2 禁止パターン（静的チェック）

- `exec` 実行前に文字列レベルで以下を検出・拒否:
  - `import`, `__import__`, `eval`, `exec`, `open`, `os`, `sys`, `subprocess`
  - `__class__`, `__subclasses__`, `__globals__` （属性アクセスによるサンドボックス脱出防止）
- **重要な制約**: 静的チェック + 制限付き `exec` は研究用途のベストエフォート防御であり、完全なセキュリティ境界ではない。本番運用で第三者コードを実行する場合は、コンテナ/VM/nsjail などOSレベル隔離を必須とする。

### 5.3 性能見積もり

| 方式 | 1世代あたりのオーバーヘッド (200グラフ × 20候補) |
|------|------------------------------------------------|
| subprocess.run() per call | ~400秒（プロセス起動） |
| **multiprocessing.Pool** | **~2秒（関数呼び出しのみ）** |

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

`phase1_summary.json` は `schema_version=3` を採用する。`report` コマンドは旧スキーマ入力にも耐性を持つ。

### グラフ生成シード

- データセット生成時のマスターシードを記録
- 各グラフの個別シードもログに含める

### チェックポイント / レジューム

世代ごとに島の状態をJSON形式で永続化し、クラッシュ・OOM等からの復旧を可能にする:

- **保存タイミング**: 各世代の評価完了後
- **保存内容**: 各島のknowledge_base（候補リスト・スコア）、現在の世代番号、早期停止カウンタ、乱数状態
- **保存先**: `checkpoints/{experiment_id}/gen_{N}.json`
- **レジューム**: `uv run python -m graph_invariant.cli phase1 --config <config.json> --resume checkpoints/{experiment_id}/gen_{N}.json` で指定世代から再開
- **ローテーション**: 直近3世代分のチェックポイントを保持し、古いものは自動削除

---

## 8. モジュール構成

```
graph_invariant/
├── REVIEW.md
├── SPEC.md
├── Research_Plan_Graph_Invariant_Discovery.md
├── pyproject.toml          # uv管理。PySR依存のためJuliaランタイムが必要（setup手順は下記参照）
├── src/
│   └── graph_invariant/
│       ├── __init__.py
│       ├── cli.py                 # フェーズ実行/レポート出力CLI
│       ├── config.py              # 全設定値（データサイズ、世代数、温度等）
│       ├── data.py                # グラフ生成・データセット管理
│       ├── llm_ollama.py          # Ollama呼び出し・コード抽出
│       ├── sandbox.py             # multiprocessing.Pool隔離実行・安全性チェック
│       ├── scoring.py             # 相関/簡潔性/新規性/総合スコア
│       ├── known_invariants.py    # 既知指標計算（新規性評価用）
│       ├── evolution.py           # 島モデル移住ロジック
│       ├── logging_io.py          # JSONL/チェックポイントI/O
│       ├── types.py               # 共通データ型
│       └── baselines/
│           ├── pysr_baseline.py   # PySRベースライン比較（シンボリック回帰）
│           └── stat_baselines.py  # 統計ベースライン（RandomForest, 線形回帰）
└── tests/
    ├── test_cli.py
    ├── test_config.py
    ├── test_data.py
    ├── test_sandbox.py
    ├── test_scoring.py
    └── ...
```

### 8.1 環境セットアップ

```bash
# Python環境（uv管理）
uv sync

# PySRベースラインにはJuliaランタイムが必要
# juliaup (推奨) または公式インストーラで Julia 1.10+ をインストール
curl -fsSL https://install.julialang.org | sh
# PySR初回実行時にJuliaパッケージが自動インストールされる（数分かかる）
```

### 8.2 CLI実行コマンド

```bash
uv run python -m graph_invariant.cli phase1 --config <config.json>
```

runs Phase 1 using the given config

```bash
uv run python -m graph_invariant.cli report --artifacts <artifacts_dir>
```

renders a markdown report from the specified artifacts directory

```bash
uv run python -m graph_invariant.cli benchmark --config <config.json>
```

runs a deterministic multi-seed Phase 1 benchmark sweep and aggregates results

---

## 9. Phase 1 ターゲットと成功基準

- **ターゲット**: `average_shortest_path_length`
- **成功基準**: Validation セットでスピアマン相関 ≥ 0.85（かつ PySR と同等以上）
- **Phase 2 への移行条件**: 上記を満たし、Test セットでも ≥ 0.80

---

## 10. 決定事項（Phase 1完了に基づき確定）

以下の項目は Phase 1 の実験完了に基づき確定した。

### 10.1 OOD検証用データセット — 確定

**合成OOD**（`src/graph_invariant/ood_validation.py` で実装済み）:
- `large_random`: 訓練分布と同じ5タイプ、$n \in [200, 500]$（100グラフ）
- `extreme_params`: 極端な密度・次数分布、$n \in [50, 200]$（50グラフ）
- `special_topology`: 決定論的構造（barbell, grid, ladder, circulant, Petersen）+ NetworkX built-in（karate, les_miserables, florentine, davis_southern_women）

**根拠**: 合成OODデータセットはスケーラビリティ・頑健性・特殊構造への汎化を系統的に検証するのに十分であり、再現性も確保できる。実世界グラフは属性データを含むため、ラベルなしグラフ不変量の検証には合成OODの方が適切。

### 10.2 Gemini API フォールバック — 該当なし

全実験はローカル `gpt-oss:20b`（Ollama経由）で実行。Gemini API フォールバックは発動していない。外部API依存なしで全パイプラインが完結することを確認。

### 10.3 Phase 2 ターゲット — algebraic_connectivity で確定

`algebraic_connectivity`（Fiedler値）を Phase 2 の主要ターゲットとして確定。理由:
- Phase 1 の Experiment 2 で algebraic connectivity への適用可能性を実証
- スペクトル理論との接点があり、数学的に興味深い発見の可能性が高い
- ASPL とは異なる数学的直感を要求するため、手法の汎用性検証に最適

### 10.4 論文投稿先 — Main conference (NeurIPS/ICML/ICLR)

Main conference（NeurIPS 2025 が第一候補）に投稿する。理由:
- 手法の新規性（LLM + island model + MAP-Elites + self-correction + bounds mode）が十分
- 4実験構成 + OOD検証 + multi-seed benchmark の体系的評価
- オープンソースの再現可能な研究基盤を提供
- 解釈可能な記号式発見は ML + 数学コミュニティ双方に訴求力がある
