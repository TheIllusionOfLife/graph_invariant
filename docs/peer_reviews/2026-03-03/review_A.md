# NeurIPS 2025 査読（Full Paper）

- 対象論文: **Harmony-Driven Theory Discovery in Knowledge Graphs via LLM-Guided Island Search**
- 形式: NeurIPS full paper 想定のレビュー（Summary / Strengths / Weaknesses / Questions / Suggestions / Score）

---

## 1. Summary（何をした論文か）🧾
本論文は、科学領域の知識グラフ（KG）に対して **「理論発見＝KG への変異（エッジ/エンティティの追加・削除）探索」**として定式化し、

- **Harmony** という複合品質指標（**compressibility / coherence / symmetry / generativity**）を設計し（Eq.1–5）、
- LLM が **claim / justification / falsification condition** を含む構造化提案（proposal）を生成し（Section 3.3, Appendix E）、
- MAP-Elites による品質多様性アーカイブ（descriptor: simplicity × gain）と、
- 4島（refine/combine/novel を循環）＋温度差＋stagnation recovery＋migration の探索機構（Section 3.4）

で探索する枠組みを提示しています（Algorithm 1）。

実験は、手作業で作った小規模 KG（線形代数・周期表の calibration、天文・物理・材料の discovery）に加え、Wikidata 由来の Physics/Materials サブ KG を含む計 5 ドメインで行い、リンク予測指標（Hits@K, MRR）を中心に評価しています（Section 4, Table 1）。LLM はローカルの **gpt-oss:20b**（Ollama）を用い、主要結果は **10 seeds の平均±標準偏差**で報告されています（Section 4.4）。

---

## 2. Strengths（良い点）✅
1. **「トリプル予測」から「理論レベル提案」への問題設定のアップグレード**
   - claim・正当化・反証条件を必須にしており、科学用途における「説明可能な仮説生成」を強く意識しています（Section 3.3）。

2. **Harmony 指標は分解可能で解釈しやすい**
   - 4要素が [0,1] に正規化され、重み付けも明確（Eq.1）。
   - 成分 ablation（Table 2）もあり、何が効いているかを議論できる作りになっています。

3. **探索設計（MAP-Elites + 島モデル + 失敗フィードバック）が合理的**
   - LLM 生成は局所最適・同質化しやすいので、
     - novelty / combine / refine の分業
     - 温度差
     - stagnation 時の constrained prompt
     - migration
     の組み合わせは筋が良いです（Section 3.4, Appendix E）。

4. **再現性の記述が比較的丁寧**
   - seed 群の明示、split 手順、ハイパラ表、プロンプトテンプレまで記載しています（Section 4, Appendix E/H）。

---

## 3. Weaknesses（主要懸念）⚠️
採択判断に強く効く論点を、優先度順に挙げます。

### 3.1 「目的関数」と「評価」が同系統で、外的妥当性が弱い（最重要）
- Harmony の **generativity** が DistMult の **Hits@10**（masked edge の復元）で定義されています（Eq.5）。
- 主な定量評価も、DistMult による Hits@K / MRR です（Section 4.4, Table 1）。

その結果、
- 「DistMult 的に予測しやすくなる方向」をスコアに含め、
- 「DistMult 的に当たるか」で評価している

ように見え、**“理論発見”としての外部評価（scientific validity）**が弱くなります。

特に本論文の実験では、**frequency ベースラインが全ドメインで Hits@10 最良**という結果も出ています（Table 1）。この事実は、「指標最適化が “科学的に意味のある新規理論” へつながっているのか？」という疑問を強めます。

### 3.2 効果量が小さく、ドメインによっては一貫して良くない
Table 1 では、Harmony が DistMult-alone を上回るのは **3/5 ドメイン**（materials, wikidata physics/materials）で、
- astronomy では DistMult-alone と同等
- physics（手作業 KG）では Harmony の方が悪化

しています（Table 1）。また、本文の Limitations でも差が有意でない例（p=0.45）が言及されています（Section 6 Limitations）。

NeurIPS full paper としては、
- **どの条件で効き、どの条件で効かないか**
- **失敗モードの原因**

をより明確に切り出す必要があります。

### 3.3 「理論発見」の主張に対し、直接的な真偽評価が薄い
提案は falsification condition を含みますが（Section 3.3）、
評価は主に「提案を入れた後のリンク予測性能」であり、

- **提案エッジ自体が正しいか（precision）**
- **隠した backtesting set と一致するか（再発見）**
- **文献で裏付けられるか（evidence）**

といった “発見” らしい評価が弱いです。

### 3.4 Harmony 成分の妥当性がドメイン依存に見える
- **Symmetry** は entity type ごとの outgoing edge type 分布の JS 距離を下げる設計ですが（Eq.4）、
  科学 KG ではタイプが役割分担していることが自然で、**タイプ間が似るほど良い**という前提は反論されやすいです。
- **Coherence** の triangle 一致条件（rac ∈ {rab, rbc}）は、関係合成としては粗く、意味的に強い保証にはなりにくいです（Eq.3）。
- contradicts を「密だとノイズ」と扱う点は、科学的議論における対立仮説の価値と緊張します（Section 6 でも限定的に認めています）。

---

## 4. Questions（著者への質問）❓
1. **外部評価**
   - 提案したエッジのうち、backtesting set と一致する割合（Precision@N）はどの程度でしょうか？
   - 文献検索（自動でも手動でも）で裏取り可能な割合は？

2. **評価循環（DistMult）**
   - Generativity を DistMult 以外（RotatE/ComplEx 等）に替えた場合でも、同様の傾向は出ますか？
   - あるいは Generativity を除いた Harmony（3成分）でも有効でしょうか？

3. **失敗モードの分析**
   - 手作業 Physics KG で Harmony が DistMult-alone を下回る理由は何でしょうか？
   - 提案の種類（ADD_EDGE/ADD_ENTITY…）や relation type の偏りはありますか？

4. **frequency が強い理由**
   - frequency が Hits@10 で常勝しているのは、KG が小さく関係分布が強いからだと解釈できます。
   - このベースラインに勝つために、Harmony/探索側をどう改善する方針ですか？

5. **専門家評価の詳細**
   - 評価者数、専門性、合意度（inter-annotator agreement）を追加できますか？

---

## 5. Suggestions（改善提案：優先度順）🛠️
### 5.1 “理論発見”としての外部妥当性を足す（最重要）
- **Backtesting set との直接照合**（Precision@N / Recall@N）
- **文献検索ベースのエビデンス付与**（support / contradict を添付し、証拠強度をスコア化）
- **人手評価の強化**（複数評価者＋合意度＋基準の明文化）

### 5.2 評価循環を断つ
- Generativity を別モデルで定義する / もしくは除いた Harmony で主結果を再提示
- 評価も DistMult 以外の KGC 指標（複数モデル平均など）を追加して「特定モデル依存ではない」ことを示す

### 5.3 “どこが効いたか”の因子分解
- LLM-only（Harmony なし）
- Harmony-only（候補生成をプログラム化）
- QD なし（単純な探索）
- island/temperature/stagnation の ablation

### 5.4 スケールと計算量
Wikidata サブ KG は既に入っているので良い方向です（Table 1）。
加えて、
- generations を増やしたときの収束
- KG サイズに対する wall time / ボトルネック

をもう一段見せると NeurIPS らしい説得力が増します。

---

## 6. Overall Score（採点）🧮
- **総合スコア（10点満点）: 6.0 / 10**
  - 新規性: 7/10（LLM×QD×KG変異探索×複合指標の組み合わせは面白い）
  - 技術妥当性: 6/10（指標設計は整理されているが、外部妥当性が弱い）
  - 実験の説得力: 5.5/10（効果量小、frequency に負けがち、成功条件の切り出しが不足）
  - 明確さ: 7/10（構成は読みやすく、付録も充実）
  - 再現性: 7/10（プロンプトテンプレや seed 明示が良い）

- **推薦**: Borderline / Weak Reject（外部評価が追加できれば Weak Accept へ上がり得る）
- **Confidence（5点満点）: 3.5 / 5**

---

## 7. One-line meta（総評）🧠
**「LLM で“仮説”を生成し、QD で探索する」発想は魅力的ですが、NeurIPS で通すには“理論発見”の外部妥当性（真偽・証拠・再発見率）を、DistMult 指標から独立して立証する必要があります。**


---

## 8. 総合スコアを **10/10** に近づけるための改善案（複数案）🚀

以下は「NeurIPS フルペーパーとして“強い採択”に持っていく」ための改善案です。**各案は独立でも併用でも**効きますが、特に **(A) 外部妥当性** と **(B) 評価循環の解消** を押さえると、査読者の最大の疑念を潰せます。

### 8.1 最優先：理論発見としての **外部妥当性** を“直接”示す（10/10の必須条件）🧪
1. **Backtesting（hold-out 真実）での Precision/Recall を提示**
   - 例：各ドメインで KG の一部エッジを隠し、提案エッジ上位Nのうち **何%が隠した真実に一致**したか（Precision@N）、真実のうち **何%を回収**できたか（Recall@N）。
   - 「KGC 指標が伸びた」よりも、「提案が当たった」を示す方が **“発見”の説得力**が跳ねます。

2. **文献エビデンス付与（自動/半自動）**
   - 各提案に対して、RAG（論文検索）で **supporting / contradicting 文献**を添付し、エビデンス強度（例：論文数、引用数、要約一致度）をスコア化。
   - *「KGに入れる前に、根拠を同梱する」*という運用は倫理面の懸念（誤情報混入）にも効きます。

3. **“新規仮説”の検証可能性の提示**
   - 既知の再発見（rediscovery）と、新規仮説（novel hypothesis）を **明確に分離**して報告。
   - 新規仮説には **具体的な反証実験プロトコル**（必要データ、測定量、判定条件）をセットで提示（既に falsification condition を要求している強みを最大化）。

---

### 8.2 最優先：評価の循環（DistMult）を断つ（査読で最も突かれるポイント）🧯
4. **Generativity を DistMult 依存から外す**
   - 選択肢：
     - (i) Generativity を **複数モデル平均**（RotatE/ComplEx/TransE 等）で定義
     - (ii) Generativity を **モデル非依存**に置き換え（例：held-out truth の当たり、文献エビデンス、ルール整合）
     - (iii) Generativity を外した **Harmony-3** で主結果を再提示（目的関数から DistMult 指標を排除）

5. **評価も複数モデルで報告**
   - DistMult 以外の KGC で Hits@K / MRR を併記し、改善が特定モデルの“癖”ではないことを示す。
   - これだけで「手法が一般的に有用」という主張が通りやすくなります。

---

### 8.3 「どこが効いたか」を因子分解し、主張を強固にする（NeurIPS的に強い）🧩
6. **ベースラインを“目的に沿って”増やす（最小セット）**
   - **LLM-only**：LLM が出した提案をそのまま採用（Harmony/QD なし）
   - **Harmony-only**：候補生成をプログラム的（ランダム/ルール）にして Harmony で選別（LLM なし）
   - **QDなし探索**：MAP-Elites を外して、単純なビーム探索/ランダム探索
   - これで、「LLM」「Harmony」「QD/島」がそれぞれどれだけ寄与したかが明確になり、説得力が増します。

7. **島・温度・stagnation recovery の ablation**
   - 島モデルの必然性（diversity の寄与）を示すと、アルゴリズム設計が“飾り”ではなくなります。

---

### 8.4 スケールと頑健性：大きめKGでも回ることを示す（フルペーパーの格）📈
8. **より大きい公開KG（数千〜数万ノード）でのスケール実験**
   - 既に Wikidata サブグラフがあるので、そこをもう一段スケール。
   - 報告すべきは「性能」だけでなく：
     - wall time（LLM呼び出し、スコア計算、再学習）
     - 収束曲線（generation vs best score）
     - コストのボトルネック
   - これがあると “実運用に耐える” 印象が大きく上がります。

9. **統計：seed と KG 構築揺らぎに対するロバスト性**
   - seeds（≥10）に加え、KG のサブサンプリングやノイズ付与でも結果が保たれるかを確認。
   - 小規模KGでは特に重要視されます。

---

### 8.5 指標（Harmony）を“科学知識表現”に寄せてアップグレードする（批判されにくくする）🧠
10. **Symmetry を「同一化」ではなく「適切な差」を許容する設計へ**
   - 例：タイプ間の分布を近づけるのではなく、
     - タイプ内の一貫性
     - 役割分担（機能差）を保った上での規則性
     を測る指標にする。
   - これで「科学KGでタイプが似るのは不自然」という反論を回避できます。

11. **Coherence を関係合成の“意味”に近づける**
   - 現在の triangle 条件は粗いので、関係合成のテンプレ（例：A causes B, B causes C → A influences C）や、
     relation embeddings を用いた合成整合性（soft constraint）に拡張する。

12. **contradicts を“ノイズ”ではなく“多元的仮説の共存”として扱うオプション**
   - contradicts を一律ペナルティにせず、
     - 反証条件が明確
     - 文献が両論併記
     の場合は「仮説競合」として許容する設計を入れると、科学的表現として強くなります。

---

### 8.6 文章・主張のチューニング（採択ラインの押し上げ）✍️
13. **“Theory discovery” の主張を正確に（過大主張を避ける）**
   - 現状でも内容は良いので、表現を
     - “theory discovery” → “hypothesis/theory proposal & evaluation framework”
     のように調整しつつ、外部評価で「発見」に近づける。
   - NeurIPS は過大主張に厳しいため、ここは効きます。

14. **成功例・失敗例の分析を増やす（定量＋定性）**
   - 「なぜその提案が Harmony を上げるのか」を 4成分に分解して説明。
   - 失敗モード（physics で悪化など）の原因分析を入れる。
   - “どこで何が壊れるか”が示せると、研究として一段格が上がります。

---

## 9. 10/10 への最短ルート（実装コスト順）🛣️
1. **(低コスト/高効果)** Generativity を外した Harmony-3 で再実験＋評価を複数KGCで併記
2. **(中コスト/超高効果)** Hold-out truth に対する Precision@N を追加（理論発見の直接評価）
3. **(中コスト/高効果)** LLM-only / Harmony-only / QDなし の因子分解ベースライン
4. **(中〜高コスト/高効果)** 文献RAGでエビデンス添付（support/contradict）
5. **(高コスト/決定打)** より大きい公開KGでスケール・収束・計算量の報告
