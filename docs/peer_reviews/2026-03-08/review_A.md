# NeurIPS 査読（Full Paper）

- 対象論文: **Harmony-Driven Theory Discovery in Knowledge Graphs via LLM-Guided Island Search**
- 査読観点: NeurIPS full paper（新規性 / 技術妥当性 / 実験の説得力 / 再現性 / 影響 / 明確さ）

---

## 1. Summary（何をした論文か）🧾
本論文は、科学領域の知識グラフ（KG）に対して **「理論発見＝KG への変異（mutation）探索」**として定式化し、

1. KG の良さを測る複合指標 **Harmony**（compressibility / coherence / symmetry / generativity）を定義し、
2. LLM が **claim・justification・falsification condition** を含む構造化提案を生成し、
3. 提案を Harmony の増分で評価して **MAP-Elites** に蓄積し、
4. refine / combine / novel を分担した **island-model search** で探索する、

という枠組みを提案しています。

実験は、手作業 KG と Wikidata 由来の KG を含む **7ドメイン**で行われ、10-seed 評価、外部 KGE ベースライン（TransE / RotatE / ComplEx）、評価循環への対処としての **Harmony-3**、因子分解（LLM-only / Harmony-only / No-QD）も含めて報告されています。

---

## 2. Strengths（良い点）✅
### 2.1 問題設定が面白い
単なるリンク予測ではなく、**理論レベルの仮説提案**を目指している点は明確な新規性があります。さらに falsification condition を必須にしているため、科学用途での「もっともらしいだけの主張」を減らそうとする設計思想も良いです。

### 2.2 Harmony が分解可能で解釈しやすい
Harmony を複数成分に分け、どの要素が寄与したかを議論できる構造になっています。NeurIPS では「何が効いているか」の説明可能性が重要なので、この点は強みです。

### 2.3 探索設計が合理的
LLM による候補生成は、同質化や局所最適にはまりやすいですが、MAP-Elites と island model を組み合わせることで、多様性を保ちながら探索する狙いは納得感があります。

### 2.4 論文の改善努力が見える
10-seed、外部ベースライン、Harmony-3、因子分解、失敗モード分析、frequency が強い理由の情報理論的説明など、以前の弱点になりやすい点をかなり補強しています。

---

## 3. Weaknesses（主要懸念）⚠️
### 3.1 外部妥当性がまだ弱い
もっとも大きな問題はここです。held-out edge に対する backtesting で **Precision / Recall がすべて 0** であり、著者自身も negative-control diagnostic と位置付けています。これは「既存エッジの単なるコピーではない」ことは示せますが、同時に **提案が当たっている証拠にもなっていません**。

### 3.2 効果量が小さい
Harmony が DistMult-alone を上回るドメインはあるものの、差はかなり小さく、統計的有意差も出ていません。NeurIPS full paper としては、「どの条件で効くのか」の切り分けは良くなった一方、**強い実証結果**としてはまだ弱いです。

### 3.3 評価軸が依然として難しい
論文は evaluation circularity を意識して Harmony-3 や外部 KGE を入れていますが、中心的なストーリーは依然として **Hits@K / MRR** に寄っています。しかし本研究の主張は「理論発見」であり、リンク予測スコアだけでは価値を取り切れていません。

### 3.4 Harmony 指標の妥当性はまだドメイン依存
symmetry を intra-type consistency に改善したのは良いですが、それでも compressibility や coherence の設計は KG の性質に依存しやすく、別分野へ一般化したときの妥当性には疑問が残ります。

### 3.5 LLM の独立寄与がまだ見えにくい
因子分解は入っていますが、結果を見ると **Harmony scoring と QD archive の寄与が大きく、LLM proposer の独立した価値はまだ十分に立証されていません**。

---

## 4. Questions for the Authors（著者への質問）❓
1. backtesting が 0 だった状況で、提案の正しさ・有望さをどのように外部検証しますか？
2. 文献検索や citation grounding を組み込み、proposal ごとの evidence score を付ける予定はありますか？
3. LLM proposer を別モデル群に変えたとき、提案の安定性や多様性はどう変わりますか？
4. Harmony が効く regime を、事後分析ではなく事前予測可能な設計ガイドへ落とし込めますか？
5. ルールマイニングや path-based reasoning との比較は可能でしょうか？

---

## 5. Suggestions（改善提案）🛠️
### 5.1 最重要：外部証拠を増やす
- 提案ごとに **文献検索ベースの support / contradict evidence** を付与する
- exact match だけでなく **soft backtesting**（同義関係・近縁関係）を導入する
- 可能なら専門家評価を複数人で行い、合意度も出す

### 5.2 評価軸を複線化する
- Hits@K / MRR だけでなく、
  - explanation quality
  - falsifiability quality
  - literature consistency
  - novelty with support
  といった評価を加えると、「理論発見」の論文として一段強くなります。

### 5.3 ベースラインを増やす
- AMIE のようなルールマイニング
- path-based inference
- template-based symbolic proposer

などを追加すると、「LLM を使う意味」がもっと明瞭になります。

### 5.4 LLM の役割を強くする
今の結果だと、LLM が主役というより「候補生成器の一つ」に見えます。proposal quality の定性分析や、LLM が出した候補の独自性をもう少し強く示せると良いです。

---

## 6. Overall Evaluation（総合評価）🧮
### 6.1 スコア
- **総合スコア: 6.5 / 10**
- **推薦: Borderline / Weak Reject 寄り**
- **Confidence: 3.5 / 5**

### 6.2 内訳
1. **新規性**: 7.5 / 10  
   LLM × QD × KG mutation search の組み合わせは面白いです。

2. **技術妥当性**: 7.0 / 10  
   以前よりかなり補強されていますが、指標と評価のズレがまだ残ります。

3. **実験の説得力**: 6.0 / 10  
   因子分解や multi-seed は良いものの、効果量と外部妥当性が弱いです。

4. **明確さ**: 7.0 / 10  
   論文構成は比較的読みやすく、問題設定も理解しやすいです。

5. **再現性**: 7.0 / 10  
   split、seed、設定、ablation は丁寧です。

---

## 7. Meta Review（一言総評）🧠
**発想は面白く、論文としての完成度も上がっています。しかし NeurIPS full paper として採択を強く引き寄せるには、「生成した仮説が実際に正しい／有望である」と示す外部証拠がまだ足りません。現状は“面白い探索フレームワーク”としては評価できる一方、“理論発見”としては証明が弱い、というのが率直な判断です。**
