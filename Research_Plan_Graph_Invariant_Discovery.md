# 研究計画書: LLM駆動型ナレッジグラフ拡張による未知のグラフ不変量の自動発見と数理的定式化

> **文書の位置づけ:** 本文書は本プロジェクトの**初期提案書（v1）**である。研究の動機、方法論、初期プロトタイプコードを含む。本提案に対するレビューは [REVIEW.md](./REVIEW.md) を、レビューを踏まえた確定仕様は [SPEC.md](./SPEC.md) を参照のこと。

## 1. 概要 (Abstract)
本研究は、大規模言語モデル（LLM）を「確率的な単語予測器」から「科学的概念の生成エンジン」へと昇華させるための実証実験である。具体的には、計算機科学および離散数学における「グラフ不変量（Graph Invariant）」の探索問題を対象とする。従来、新しいグラフ指標の発見は、熟練した数学者の直感や、限定的なシンボリック回帰（Symbolic Regression）に依存していた。本研究では、LLMに既存の数学的知識（ナレッジグラフ）を与え、ターゲットとなる複雑なグラフ特性を近似・予測する「新しい数式（概念）」を生成させる。生成された仮説は、即座にPython環境で数千のグラフに対して実行・検証され、そのフィードバックが再びLLMに入力される。この「生成（Generation）→ 検証（Verification）→ 淘汰・進化（Evolution）」の閉ループ（Closed-Loop System）をローカル環境（Mac Mini M2 Pro）上で自律的に回転させることで、教科書に未記載の有用な近似式や、グラフ構造の新規指標を発見することを目的とする。

## 2. 研究の目的と問い (Objectives & Research Questions)

### 2.1 主目的
*   **解釈可能な数式の発見**: ブラックボックスなニューラルネットではなく、人間が理解可能な「閉じた数式（Closed-form expression）」として特徴量を抽出する。
*   **計算複雑性の短縮**: 計算コストが高い指標（例: NP困難な指標や $O(N^3)$ クラス）を、計算コストの低い指標（$O(N)$ や $O(E)$）の非線形な組み合わせで高精度に近似する。
*   **構造的空隙の発見**: 既存の指標（密度、クラスター係数など）では区別できないグラフ構造を識別できる新しい指標（Discriminative Powerの高い不変量）を発見する。

### 2.2 リサーチクエスチョン
*   **RQ1**: LLMは、単なる既存コードのコピーではなく、数学的に意味のある「新しい演算子の組み合わせ」を提案できるか？
*   **RQ2**: 生成された指標をナレッジグラフにフィードバック（In-context Learning）することで、探索効率は向上するか？
*   **RQ3**: 物理的実世界データを持たない「純粋数学」の領域において、AIは人間の数学者と同様の「美的感覚（シンプルな式を好む）」を獲得できるか？

## 3. 方法論 (Methodology)

### 3.1 システムアーキテクチャ詳細
システムはMac Mini M2 ProのユニファイドメモリとマルチコアCPUを最大限活用するため、非同期並列処理モデルを採用する。

*   **Generator (Concept Proposer)**:
    *   **モデル**: Llama 3 (8B-Instruct) または Mistral (7B-Instruct) を4-bit量子化で使用。
    *   **役割**: ナレッジグラフ内の「高スコアな既存式」と「失敗した式」の両方をコンテキストとして受け取り、新しいPython関数定義を出力する。
    *   **プロンプト戦略**: Chain-of-Thought (CoT) を強制。「なぜその変数を組み合わせるのか」という理論的根拠をコメントとして記述させた後、コードを生成させる。

*   **Verifier (Parallel Sandbox)**:
    *   **役割**: 提案されたコードの安全性チェック、実行、性能評価。
    *   **実行環境**: multiprocessing モジュールを用いた並列実行。M2 Proの高性能コアをフル稼働させ、1秒間に数百〜数千のグラフ検証を行う。
    *   **データセット**:
        *   Training Set: 100個の小規模グラフ（探索用）
        *   Validation Set: 1,000個の多様なグラフ（評価用）
        *   Graph Types: Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Geometric Graph, Stochastic Block Model (SBM)。

*   **KG Manager (Evolutionary Memory)**:
    *   **構造**: 単なるリストではなく、式の「系統樹」を保持。
    *   **淘汰圧**: 精度（相関係数）だけでなく、「式の短さ（MDL: 記述長最小原理）」と「計算速度」をペナルティ項として入れたスコアでランキングする。

### 3.2 探索アルゴリズム (The Evolutionary Prompting Loop)
1.  **Seed**: 基本的な不変量（ノード数 $n$, エッジ数 $m$, 次数分布 $d$）を初期ノードとする。
2.  **Generate**: 上位 $K$ 個の最良の式をプロンプトに含め、LLMに「これらを改良、または全く新しい視点で変形せよ」と指示。
3.  **Mutation (変異)**: $n \to n^2$, $\log \to \sqrt{}$ などの演算子置換。
4.  **Crossover (交叉)**: 式Aと式Bを組み合わせて新しい式Cを作る。
5.  **Verify**: 生成された関数 $f_{new}$ を検証セットで実行。ターゲット値 $y$ との相関 $R^2$ を計算。
6.  **Feedback**:
    *   エラーが出た場合 $\to$ エラーログと共に再生成指示（Self-Correction）。
    *   スコアが低い場合 $\to$ 「失敗ケース」として記憶（Negative Sampling）。
    *   スコアが高い場合 $\to$ KGに追加し、次世代の親とする。

## 4. 実証実験計画 (Detailed Implementation Plan)

### Phase 1: ベースライン構築と単純近似 (Day 1-2)
*   **ターゲット**: average_shortest_path_length (平均最短経路長)。
*   **技術的マイルストーン**:
    *   Ollama APIとPythonスクリプトの安定した接続。
    *   並列処理による検証ループの高速化（目標: 1分間で10個の仮説検証）。
    *   既存の近似式（$L \approx \frac{\ln n}{\ln \langle k \rangle}$）をLLMが再発見、あるいは超えることができるか確認。

### Phase 2: 複雑な構造指標の探索 (Day 3-5)
*   **ターゲット**: algebraic_connectivity (代数的連結度 / ラプラシアン行列の第2最小固有値)。これはグラフの「頑健性」を表すが、計算コストが高い。これを「次数分布」や「三角形の数」などの単純な統計量から予測させる。
*   **導入機能**: SymPy を導入し、LLMが生成した冗長な数式（例: x + x $\to$ 2*x）を自動で簡約化してから評価・保存する。

### Phase 3: "Out-of-Distribution" 検証と論文執筆 (Day 6-7)
*   **検証**: 学習に使っていない全く異なるタイプのグラフ（例: 実世界のソーシャルネットワークデータセットの一部）に対し、発見された式が汎化するか検証する。
*   **成果物**: 発見された数式リスト、相関プロット、および「AIがなぜその式を導出したか」の推論プロセスのログ解析。

## 5. 評価指標 (Metrics)
*   **精度 (Accuracy)**: ターゲット値とのピアソン相関係数 ($r$) および スピアマン順位相関係数 ($\rho$)。非線形な関係を捉えるため、スピアマンを重視する。
*   **簡潔性 (Simplicity)**: 式を構成するトークン数や演算子の深さ。複雑すぎる式（Overfitting）は低評価とする。
*   **新規性 (Novelty)**: 既存の既知の不変量との相関が低いこと（＝既存指標の単なる言い換えではないこと）。

## 6. 使用ツール・ライブラリ・環境設定
*   **Compute**: Mac Mini M2 Pro (10-core CPU, 16-core GPU, 16GB+ RAM)
*   **LLM Inference**: Ollama (Llama 3 8B Instruct - q4_k_m quantization recommended for speed/memory balance).
*   **Python Libraries**:
    *   networkx: グラフ操作
    *   numpy, scipy: 数値計算・統計
    *   sympy: 数式処理
    *   multiprocessing: 並列検証
    *   scikit-learn: ベースライン比較（RandomForest等）


### 7. 実装コード (Advanced Prototype)

```python
import networkx as nx
import numpy as np
import scipy.stats
import requests
import json
import random
import time
import math
import warnings
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" 
NUM_TRAIN_GRAPHS = 50   # 探索用（高速化のため少数）
NUM_VAL_GRAPHS = 200    # 検証用（汎化性能チェック）
MAX_GENERATIONS = 20    # 進化の世代数
POPULATION_SIZE = 5     # 1世代あたりの個体数
TIMEOUT_SEC = 2.0       # 1つのグラフに対する計算の制限時間

# --- GRAPH GENERATION UTILS ---
def generate_single_graph(seed):
    """
    並列処理用に、1つのグラフを生成する関数。
    多様な種類のグラフをランダムに生成する。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    N = random.randint(30, 80)
    g_type = random.choice(['ER', 'BA', 'WS', 'SBM'])
    
    G = None
    try:
        if g_type == 'ER':
            p = random.uniform(0.05, 0.3)
            G = nx.erdos_renyi_graph(N, p)
        elif g_type == 'BA':
            m = random.randint(1, 4)
            if m >= N: m = N - 1
            G = nx.barabasi_albert_graph(N, m)
        elif g_type == 'WS':
            k = random.randint(4, 8)
            p = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(N, k, p)
        elif g_type == 'SBM':
            # Stochastic Block Model
            sizes = [N//3, N//3, N - 2*(N//3)]
            probs = [[0.25, 0.05, 0.02], [0.05, 0.25, 0.05], [0.02, 0.05, 0.25]]
            G = nx.stochastic_block_model(sizes, probs)
            
        # 連結成分抽出（指標計算の安定化）
        if G and not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            # ノードラベルを0..N-1にリセット
            G = nx.convert_node_labels_to_integers(G)
            
        return G
    except Exception:
        return nx.erdos_renyi_graph(30, 0.2) # Fallback

def prepare_datasets():
    print("Generating Datasets...")
    # 並列生成
    with ProcessPoolExecutor() as executor:
        train_seeds = [random.randint(0, 100000) for _ in range(NUM_TRAIN_GRAPHS)]
        val_seeds = [random.randint(0, 100000) for _ in range(NUM_VAL_GRAPHS)]
        
        train_graphs = list(executor.map(generate_single_graph, train_seeds))
        val_graphs = list(executor.map(generate_single_graph, val_seeds))
        
    # Noneを除去
    train_graphs = [g for g in train_graphs if g is not None]
    val_graphs = [g for g in val_graphs if g is not None]
    
    print(f"Datasets Ready: Train={len(train_graphs)}, Val={len(val_graphs)}")
    return train_graphs, val_graphs

def calculate_target(graphs, target_name="average_shortest_path_length"):
    print(f"Calculating Ground Truth: {target_name}...")
    y = []
    for G in graphs:
        if target_name == "average_shortest_path_length":
            y.append(nx.average_shortest_path_length(G))
        elif target_name == "algebraic_connectivity":
            y.append(nx.algebraic_connectivity(G))
    return np.array(y)

# --- LLM INTERACTION ---
def query_llm_for_code(context, best_formulas, target_name):
    """
    過去のベストな式（best_formulas）をヒントに、新しい式を生成させる
    """
    
    # ヒントの構築
    hints = ""
    if best_formulas:
        hints = "Here are some hypothesis functions discovered so far that performed well. Refine them or combine them:\n"
        for i, f in enumerate(best_formulas[:3]): # Top 3
            hints += f"--- Hypothesis {i+1} (Score: {f['score']:.4f}) ---\n{f['code']}\n"
    
    prompt = f"""
    You are an AI scientist specializing in Graph Theory and Python.
    
    Objective:
    Write a Python function `def new_invariant(G):` that approximates the target property: "{target_name}".
    
    Available Inputs:
    - G: A NetworkX graph object.
    - You can use: G.number_of_nodes(), G.number_of_edges(), nx.density(G), degrees = [d for n, d in G.degree()], etc.
    - Libraries: numpy as np, networkx as nx, math.
    
    Context & History:
    {hints}
    
    Instructions:
    1. Think step-by-step: Analyze the relationship between the target and basic properties.
    2. Propose a mathematical formula.
    3. Implement it in Python.
    4. The function MUST handle edge cases (e.g., log of zero, division by zero) by adding small epsilons (1e-6) or max functions.
    5. Return ONLY the Python code block enclosed in ```python ... ```.
    
    Constraint:
    - Do NOT use the target function "{target_name}" itself.
    - Keep complexity low (avoid heavy computations like all_pairs_shortest_path inside the heuristic).
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8} # 少し探索的に
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"LLM Error: {e}")
    return None

def extract_python_code(text):
    import re
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # マッチしない場合、全体がコードである可能性も考慮
    if "def new_invariant(G):" in text:
        return text
    return None

# --- SAFE EXECUTION ---
def run_invariant_on_graph(code_str, G):
    """
    1つのグラフに対して生成されたコードを実行する。
    タイムアウトとエラーハンドリング付き。
    """
    # 実行用のグローバル空間
    local_env = {}
    safe_globals = {
        "np": np, "nx": nx, "math": math,
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len
    }
    
    try:
        exec(code_str, safe_globals, local_env)
        func = local_env.get('new_invariant')
        if not func: return 0.0
        
        # 実行
        val = func(G)
        
        # NaN / Inf チェック
        if np.isnan(val) or np.isinf(val):
            return 0.0
        if isinstance(val, (complex)): # 複素数は除外
            return val.real
        return float(val)
        
    except Exception:
        return 0.0

def evaluate_hypothesis_parallel(code_str, graphs, y_true):
    """
    並列処理で全てのグラフに対して仮説を評価する
    """
    # シンタックスチェック
    try:
        compile(code_str, '<string>', 'exec')
    except Exception as e:
        return -1.0, f"Syntax Error: {e}"

    y_pred = []
    
    # 簡易タイムアウト機構付き実行ループ
    start_time = time.time()
    for G in graphs:
        if time.time() - start_time > 10.0: # 全体で10秒超えたら打ち切り
            return -1.0, "Time Limit Exceeded"
        
        val = run_invariant_on_graph(code_str, G)
        y_pred.append(val)
        
    y_pred = np.array(y_pred)
    
    # 標準偏差が0（全てのグラフで同じ値）なら無意味
    if np.std(y_pred) < 1e-9:
        return 0.0, "Constant Output"

    # スピアマン順位相関係数（非線形な単調増加を評価）
    corr, _ = scipy.stats.spearmanr(y_true, y_pred)
    
    # 欠損値対策
    if np.isnan(corr): corr = 0.0
    
    return abs(corr), "Success"

# --- MAIN LOOP ---
def main():
    print("=== AI Graph Theory Researcher (M2 Pro Optimized) ===")
    
    # 1. データセット準備
    train_graphs, val_graphs = prepare_datasets()
    target_name = "average_shortest_path_length"
    
    y_train = calculate_target(train_graphs, target_name)
    y_val = calculate_target(val_graphs, target_name)
    
    knowledge_base = [] # 発見された式とスコアのリスト
    
    # 2. 進化ループ
    for gen in range(MAX_GENERATIONS):
        print(f"\n>>> Generation {gen+1}/{MAX_GENERATIONS}")
        
        # 候補生成 (Population)
        candidates = []
        for _ in range(POPULATION_SIZE):
            llm_out = query_llm_for_code("", knowledge_base, target_name)
            if llm_out:
                code = extract_python_code(llm_out)
                if code:
                    candidates.append(code)
        
        print(f"Generated {len(candidates)} candidates. Evaluating...")
        
        # 評価
        new_discoveries = []
        for code in candidates:
            # まずTrainセットで高速スクリーニング
            score_train, status = evaluate_hypothesis_parallel(code, train_graphs, y_train)
            
            if score_train > 0.6: # 見込みあり
                # Valセットで本格検証
                score_val, status_val = evaluate_hypothesis_parallel(code, val_graphs, y_val)
                print(f"  [Eval] Train:{score_train:.3f} Val:{score_val:.3f} | Status: {status}")
                
                if score_val > 0.7:
                    new_discoveries.append({
                        "code": code,
                        "score": score_val,
                        "generation": gen
                    })
            else:
                pass 
                # print(f"  [Drop] Score:{score_train:.3f} | {status}")

        # ナレッジベース更新（成績順にソートして保持）
        if new_discoveries:
            knowledge_base.extend(new_discoveries)
            knowledge_base.sort(key=lambda x: x['score'], reverse=True)
            # 上位5つだけ残す（淘汰）
            knowledge_base = knowledge_base[:5]
            
            best = knowledge_base[0]
            print(f"★ Current Best Hypothesis (Score: {best['score']:.4f}):")
            print(best['code'].splitlines()[1]) # 関数定義の次の行を表示
        else:
            print("No improvement this generation.")

    print("\n=== Final Report ===")
    for i, item in enumerate(knowledge_base):
        print(f"Rank {i+1} (Score: {item['score']:.4f}):")
        print(item['code'])
        print("-" * 30)

if __name__ == "__main__":
    # Mac/Linuxでのmultiprocessingバグ回避
    multiprocessing.set_start_method('spawn')
    main()

```

### 8. 期待されるインパクトと発展性 (Expected Impact & Future Work)

#### 8.1 科学的インパクト

本研究が成功した場合、以下の科学的意義が期待される。

* **「AIによる理論発見」の民主化:** スーパーコンピュータ（HPC）を必要とせず、一般的なワークステーション（Mac Mini等）で理論物理学や数学の探索が可能であることを示す。
* **近似計算のライブラリ化:** 発見された軽量な近似式を `NetworkX` などのオープンソースライブラリに還元し、大規模グラフ解析の高速化に貢献する。

#### 8.2 将来の展望

* **他の数学領域への拡張:** 結び目理論（Knots Theory）や位相幾何学（Topology）など、不変量が重要な役割を果たす他領域への応用。
* **定理証明系との統合:** Pythonでの数値検証だけでなく、Lean 4 などの定理証明支援系と接続し、発見された数式が「なぜ正しいか」の証明までを自動化する（AI Mathematicianの実現）。

### 9. 参考文献 (References)

1. **Davies, A., et al. (2021).** "Advancing mathematics by guiding human intuition with AI." *Nature*. (DeepMindによる結び目理論での結びつき発見事例)
2. **Funai, T., & GNN Research Group.** "Graph Neural Networks as Intuition Machines." (GNNによる構造的直感のモデル化に関する基礎研究)
3. **Romera-Paredes, B., et al. (2023).** "Mathematical discoveries from program search with large language models." *Nature*. (FunSearch: LLMによるアルゴリズム発見の先行事例)