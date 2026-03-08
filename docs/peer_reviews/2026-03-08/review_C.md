# NeurIPS 2026 Peer Review (Third Revision)

## Paper: "Harmony-Driven Theory Discovery in Knowledge Graphs via LLM-Guided Island Search"

---

## Summary

本論文は、科学的知識グラフ（KG）における理論発見を自動化するフレームワーク「Harmony」を提案する。Harmonyスコアは圧縮性・整合性・対称性・生成性の4成分（+オプションのfrequency成分）からなる複合品質指標であり、LLMが生成するKG変異を評価する。第3版では、外部KGEベースライン4種（DistMult, TransE, RotatE, ComplEx）の追加、因子分解実験、frequency優位性の情報理論的分析、評価循環性への対処（Harmony-3）、対称性成分のintra-type再設計、backtesting、regime characterisation、統計的有意性検定（paired bootstrap CI + Cliff's delta）など、大幅な拡充が行われた。7ドメイン（手作り5 + Wikidata 2）で10シード評価を実施している。

---

## Overall Assessment

**Recommendation: Weak Reject (5/10)**

第3版は実験設計・分析の充実度において前版から顕著に改善されており、NeurIPSレベルの実験方法論に近づいている。特に因子分解実験、frequency分析、評価循環性への対処、統計的検定の導入は高く評価する。しかし、いくつかの新たな深刻な問題が浮上している。第一に、Figure 1とFigure 2が本論文のKG実験とは無関係な内容（PySR, Linear Regression, Spearman Correlation等）を含んでおり、提出物の品質管理に重大な懸念がある。第二に、backtestingが全ドメインで完全にゼロ（Table 3）という結果は、提案がheld-outエッジを一切回復できないことを意味し、手法の実用的価値に疑問を投げかける。第三に、因子分解の結果（Table 4）は、LLM proposerではなくMAP-Elitesアーカイブが主要な貢献者であることを示しており、論文の中心的主張であるLLM-guided discoveryの意義が弱まっている。

前版と同スコア（5/10）とした理由は、実験方法論の改善（+1相当）と新たに浮上した問題（Figure混入、backtesting全ゼロ、LLM寄与の弱さ）による相殺である。

---

## Strengths

### S1: 実験方法論の大幅な成熟

本改訂で最も印象的なのは、実験設計の厳密化である。paired bootstrap CI（B=2000）とCliff's delta効果量（Appendix H）の導入、Harmony-3（δ=0）による評価循環性への対処、因子分解実験（LLM-only, Harmony-only, No-QD）によるコンポーネント寄与の分離、およびfrequency優位性の情報理論的分析（Section 5.5, Appendix I）は、いずれもNeurIPSレベルの実験基準に合致している。特に、全てのp値が非有意であることを率直に報告している（Table 1, p=0.38–1.00）点は、研究の誠実さとして高く評価する。

### S2: Frequency分析の深さ

Section 5.5のfrequency分析は本論文の最も価値ある知見の一つである。エッジタイプ分布のShannon entropyとfrequency Hits@10の負の相関を示し、小規模KGにおけるfrequencyの構造的優位性を情報理論的に説明している。Hits@K ≥ p_maxという下界の導出は簡潔だが洞察的である。Harmony+Frequencyハイブリッド（εパラメータ）の感度分析も、実践的な指針を提供している。

### S3: 因子分解による寄与の透明化

Table 4の因子分解実験は、パイプラインの各要素がどの程度貢献しているかを明確にしている。Harmony-onlyがNo-QDを全5ドメインで上回るという結果は、MAP-Elitesの品質-多様性アーカイブが提案品質の主要なドライバーであることを実証している。この透明性は、読者が手法の価値を正確に評価するために不可欠である。

### S4: 対称性成分の改良

Eq. 4-5のintra-type behavioural consistencyへの再設計は、前版で指摘した「機能的に異なるエンティティタイプ間の対称性を不当に要求する」問題に直接対処している。エンティティごとのJS距離をtype centroidから計算し、エンティティ数で重み付け平均する設計は概念的に適切であり、outgoing edgeがないエンティティの除外やsingle-entity typeの取り扱いも明示されている。

### S5: データセット統計の大幅改善

Table 6で7ドメイン全てのentity/edge/type countとソースを統一的に報告している点は再現性に大きく貢献する。特に注目すべきは、手作りKGの規模が前版（17–22エンティティ）から大幅に拡大されている点である（Linear algebra: 50エンティティ/75エッジ、Periodic table: 153エンティティ/326エッジ）。これはKGの実質的な拡充を意味する。

---

## Weaknesses

### W1: Figure 1とFigure 2が論文内容と不整合（Critical）

Figure 1はキャプションで「Hits@10 comparison across discovery domains」と述べているが、実際の図にはPySR、Linear Regression、Random Forest、「Spearman Correlation」、「Method Comparison」「Validation/Test」等のラベルが含まれており、KGリンク予測の結果ではなく、別の実験（おそらくmatrix/graph property予測実験）の結果が混入している。同様にFigure 2も「Convergence of valid proposal rate」と銘打ちつつ、「Best Score」「Archive Coverage」等の全く異なるメトリクスを含む大量のサブプロットが表示されており、本論文のKG実験とは無関係に見える。

これは提出物の品質管理における重大な問題であり、以下の懸念を生む。（a）正しい図が存在するが差し替えを忘れたのか、（b）本論文の実験が実際にはFigureに対応する別の実験系列の一部なのか判断できない。いずれにせよ、査読者が結果を視覚的に検証できないため、Table 1の数値の信頼性にも影響する。

### W2: Backtestingが全ドメインで完全ゼロ（Major）

Table 3の結果は、top-5/10/20の提案がheld-outエッジと一切マッチしないことを示している。著者はこれを「design intent（提案は既存トリプルをコピーする必要がない）と互換」と解釈しているが、この解釈には問題がある。

まず、backtestingは著者自身がSection 4.5で「external validity evidence independent of the training objective」として導入したものであり、全ゼロという結果はexternal validityの証拠が得られなかったことを意味する。「negative-control diagnostic」への解釈の後退は、結果が期待に反したことを示唆している。次に、exact-matchが厳しすぎるという議論は理解できるが、relaxed matching（例：source entityとedge typeのみ一致、またはembedding空間での近傍一致）を試みて部分的な妥当性を示すべきであった。さらに、全ゼロのbacktestingを「新規性の証拠でもない」と認めている以上、このセクションが論文に追加的価値をもたらしているか疑問である。

### W3: 因子分解がLLMの寄与を弱体化（Major）

Table 4の因子分解結果は、意図せずして論文の中心的主張を損なっている。

Harmony-only（ランダム提案者、LLMなし）がNo-QD（LLM+Harmony、MAP-Elitesなし）を全5ドメインで上回る。これは、MAP-Elitesアーカイブが最も重要なコンポーネントであり、LLM proposerの独立した寄与は確認できないことを意味する。LLM-onlyはHarmonyスコアリングなしでは提案の品質区別ができない（gain=1.0に飽和）ため、LLMが有用な提案を生成しているかどうかすら評価できない。

論文のタイトルは「LLM-Guided Island Search」であり、Abstractでも「An LLM proposer generates candidate theory-level propositions」が中心的貢献として位置づけられている。しかし因子分解は、LLMの寄与がMAP-Elitesとランダム提案の組み合わせで代替可能である可能性を示唆している。LLM proposer vs. random proposer（いずれもHarmonyスコアリング+MAP-Elites付き）の直接比較が欠如しているため、LLMの独立的価値が検証されていない。

### W4: 統計的有意性の不在が持続（Major）

10シード評価とpaired bootstrap CIの導入にもかかわらず、Harmony vs. DistMult-aloneの差は全5ドメインで統計的に非有意である（p=0.38–1.00、Cliff's δ_C ≤ 0.12）。最も有望なWikidata Materials（p=0.38, δ_C=+0.12）ですら「小さい」効果量の閾値（0.33）に達していない。

3回の改訂を経てなお統計的有意性が達成されないことは、Harmony→DistMult改善という仮説がこの実験規模では検証不能である、あるいは真の効果が存在しない可能性を示唆している。著者はSection 6で「larger-scale experiments... may be needed」と述べているが、3回の改訂で規模拡大が十分に行われていない。

### W5: Expert Rubricの削除（Major）

前版（v2）にはSection 5.5としてExpert Rubric評価（plausibility 3.4, novelty 3.1, falsifiability 3.6）が含まれていたが、本版ではこのセクションが完全に削除されている。前版の査読で「Expert Rubricの詳細不足」を指摘したが、これは「詳細を追加せよ」という意味であり「削除せよ」という意味ではない。

「理論発見」フレームワークにとって、提案の科学的質の評価は不可欠である。リンク予測メトリクス（Hits@10, MRR）は構造的有用性を測るが、提案のplausibility、novelty、falsifiabilityといった科学的価値は別途評価する必要がある。Expert Rubricの削除により、Harmonyの「理論発見」としての価値を評価する手段が論文から完全に失われた。

### W6: Table 6のデータセット統計と本文の不整合（Moderate）

Table 6では手作りKGの規模が前版から大幅に変更されている（例：Linear algebra: 17→50エンティティ、Periodic table: 22→153エンティティ、Astronomy: 20→41エンティティ）。しかし本文のSection 4.1では依然として「17 entities (matrix, vector, eigenvalue...）」「22 entities (chemical elements...）」と旧版の記述が残っている箇所がある。

さらに重要なのは、この規模変更がTable 1の結果とどう関係するかが不明瞭な点である。Table 1の結果は新しいKG（50エンティティ版等）で計算されたのか、旧版（17エンティティ版）で計算されたのか。もし新しいKGであるなら、前版のTable 1と比較してHits@10やMRRの値がほぼ同一であることが不自然である（例：Astronomy Harmony Hits@10 = 0.24±0.17は前版と全く同一）。

### W7: Appendix K（Rebuttal-Oriented Supplementary Tables）の内容が本論文と無関係（Moderate）

Table 7–11は「NeurIPS matrix runs」「PySR budget」「self-correction failure breakdown」「ablation_sc_off」等、本論文のHarmony KG実験とは異なる実験系列に関するものである。Figure 1–2の問題と合わせて、本論文が複数の実験プロジェクトの成果物を混合している可能性がある。査読対象の論文として、関連する実験結果のみを含むべきである。

---

## Questions for Authors

1. **Figure 1–2の正体**: 正しいFigure（KGドメインごとのHits@10棒グラフ、validity rate収束曲線）は存在するか？差し替えが可能か？
2. **Table 6とSection 4.1の不整合**: Table 1の結果は50エンティティ版Linear Algebraで計算されたのか、17エンティティ版か？前版と値が同一である理由は？
3. **LLM vs. Random proposerの直接比較**: Harmony-only（random proposer + Harmony + MAP-Elites）とFull pipeline（LLM + Harmony + MAP-Elites）の直接比較は実施したか？LLMの独立的寄与を示すデータはあるか？
4. **Relaxed backtesting**: exact-matchではなくrelaxed matching（entity-type level, embedding近傍等）でのbacktesting結果はあるか？
5. **Expert Rubric削除の理由**: 前版のExpert Rubric評価を削除した理由は何か？評価者確保の問題か、結果が好ましくなかったか？
6. **Appendix K, Table 7–11の関連性**: これらのテーブルは本論文のHarmony KG実験とどう関係するか？

---

## Comparison with Previous Versions

| 項目 | v1 (4/10) | v2 (5/10) | v3 (5/10) | 改善度 |
|---|---|---|---|---|
| 評価シード数 | 1 | 10 (mean±std) | 10 (mean±std) + paired bootstrap CI | ✓✓ |
| KGEベースライン | DistMultのみ | DistMultのみ | +TransE, RotatE, ComplEx | ✓✓ |
| 評価循環性対処 | なし | なし | Harmony-3 (δ=0) | ✓ |
| 因子分解 | なし | なし | LLM-only, Harmony-only, No-QD | ✓✓ |
| Frequency分析 | なし | 定性的言及のみ | 情報理論的分析 + hybrid | ✓✓ |
| Backtesting | なし | なし | 全ドメインゼロ | △ (試みたが結果なし) |
| 統計的検定 | なし | なし | Bootstrap CI + Cliff's delta | ✓✓ |
| 対称性の設計 | inter-type | inter-type | intra-type再設計 | ✓ |
| データセット統計 | 不完全 | 不完全 | Table 6統一 | ✓ |
| Regime characterisation | なし | なし | Section 5.9 | ✓ |
| Expert Rubric | 詳細不足 | 詳細不足 | **削除** | ✗✗ |
| Figure正確性 | v1不整合 | v2修正済 | **再び不整合** | ✗✗ |
| Appendix整合性 | — | — | 無関係テーブル混入 | ✗ |

---

## Detailed Assessment by Criterion

### Soundness (2.5/4)

実験方法論は大幅に改善された（10シード、bootstrap CI、Cliff's delta、因子分解、複数KGEベースライン）。しかし、Figure 1–2の混入は結果の視覚的検証を不可能にしており、Table 6とSection 4.1の数値不整合はデータの信頼性に疑問を投げかける。統計的有意性が全ドメインで未達成である点も依然として重要な懸念である。

### Contribution (2/4)

因子分解の結果、MAP-Elitesアーカイブが主要貢献者でありLLMの独立的寄与が不明確となった。Harmonyメトリクス自体の設計は興味深いが、downstream（リンク予測）での有効性が統計的に示されていない。Frequency分析とregime characterisationは知見として価値があるが、これらは「Harmonyが有効でない理由の分析」であり「Harmonyの有効性の証拠」ではない。Expert Rubricの削除により、「理論発見」としての固有の価値を示す手段が失われた。

### Clarity (2.5/4)

本文の記述自体は前版から改善されているが、Figure 1–2の内容不整合とAppendix Kの無関係テーブル混入により、論文全体の品質管理に疑問が生じている。本文のSection 4.1の記述とTable 6の数値の不一致も読者を混乱させる。

### Significance (2/4)

問題設定（KG上の理論発見）は引き続き興味深い。しかし、Harmonyが既存手法（frequency, TransE）に対して有意な改善を示せないこと、backtestingが全ゼロであること、およびLLMの独立的寄与が不明確であることから、現時点での実践的・学術的インパクトは限定的である。

---

## Suggestions for Next Revision

### 最優先（提出前に必ず修正すべき事項）

**Figure差し替え**: Figure 1–2を本論文のKG実験に対応する正しい図に差し替える。これは論文の信頼性に直結する最重要事項である。

**Appendix Kの整理**: Table 7–11のうち本論文に無関係なテーブルを削除するか、関連性を明示する。

**Section 4.1とTable 6の整合**: 本文中のエンティティ数記述をTable 6に合わせて更新する。

### 高優先（採択水準到達に必要）

**LLM vs. Random proposerの直接比較**: Full（LLM + Harmony + MAP-Elites）vs. Harmony-only（Random + Harmony + MAP-Elites）の比較により、LLMの独立的寄与を定量化する。もしLLMが有意に寄与しないなら、論文のframing（"LLM-Guided"）を「Quality-Diversity-Driven」等に変更すべきである。

**Expert Rubricの復活と拡充**: 3名以上の独立した専門家による評価、inter-rater agreement、個別スコアの報告を含める。「理論発見」フレームワークの固有価値を示すために不可欠。

**Relaxed backtesting**: entity-type level matching、embedding空間での近傍matching等のrelaxed基準でbacktestingを再実施し、部分的な外部妥当性を検証する。

**大規模KGでの評価**: FB15k-237やCoDEx等の標準ベンチマークでの少なくとも1ドメインの追加評価。Wikidata KG（253エンティティ）は前版からの改善だが、標準ベンチマークとの比較可能性がない。

### 中優先（論文の完成度向上）

Harmony-downstream相関分析（Section 5.8で言及されているが、具体的な相関係数の報告が見当たらない）の結果を明示的に報告する。ハイブリッドHarmony+Frequencyの詳細な結果テーブルをAppendixに含める。世代数（Tmax）を増やした場合の収束実験を追加する。

---

## Verdict

第3版は実験方法論において顕著な進歩を遂げており、著者の改訂努力は明白である。因子分解、frequency分析、統計的検定、KGEベースライン追加、対称性再設計はいずれも論文の質を高めている。

しかし、本改訂は「改善しながら新しい問題を生む」というパターンに陥っている。Figure 1–2の無関係な図の混入は提出物の品質管理として看過できない。backtesting全ゼロは外部妥当性の証拠がないことを意味する。因子分解はLLMの独立的寄与を確認できなかった。Expert Rubricの削除は「理論発見」としての固有価値の評価手段を失わせた。統計的有意性は3回の改訂を経てなお未達成である。

現状のHarmonyフレームワークは、そのarchitectural design（MAP-Elites + Harmonyメトリクス + LLM proposer）は興味深いが、各コンポーネントの寄与が期待通りではない段階にある。最も大きな貢献はMAP-Elitesアーカイブにあり、LLMの寄与は不明確、Harmonyメトリクスのdownstream効果は統計的に示されていない。

投稿先としては、NeurIPS main trackよりも、ALife、GECCO（Quality-Diversity track）、AAAI（Knowledge Representation track）、またはNeurIPS Workshop（例：AI4Science）がより適切と考える。特にGECCOのQuality-Diversity trackは、MAP-Elitesとisland-modelの寄与を前面に出した場合に良いfitが得られるだろう。

---

## Scores

| Criterion | Score | (前版比) |
|---|---|---|
| Soundness | 2.5/4 | (↑ from 2) |
| Contribution | 2/4 | (→) |
| Clarity | 2.5/4 | (↓ from 3, Figure/Appendix混入のため) |
| Significance | 2/4 | (→) |
| **Overall** | **5/10** | **(→)** |
| **Confidence** | **4/5** | (→) |

### スコア据え置きの根拠

実験方法論の改善（Soundness +0.5）は評価するが、Figure/Appendixの整合性問題（Clarity −0.5）、Expert Rubric削除、backtesting全ゼロ、LLM寄与の弱体化という新たな問題が相殺し、全体スコアは前版と同じ5/10に留まる。根本的な課題（統計的有意な改善の不在）が未解決である限り、スコアの大幅な向上は困難である。
