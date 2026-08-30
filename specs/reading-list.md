[← specs index](README.md)

# Reading list

Grouped by the phase that needs them. Everything below was verified to exist; read the abstracts before
committing to a phase.

## Benchmark and task
- **[T1]** *Overview of the TREC Clinical Trials Track* (2021 / 2022 / 2023), NIST — task definition,
  qrels semantics, official measures, pooling. Start here.
  <https://www.trec-cds.org/2022.html>
- **[T2]** Zhuang, Koopman & Zuccon, *Team IELAB at TREC Clinical Trial Track 2023: Enhancing Clinical
  Trial Retrieval with Neural Rankers and Large Language Models* (2024) — LLM-generated synthetic patient
  descriptions as training data for dense and sparse retrievers on this collection.
  <https://arxiv.org/abs/2401.01566>
- **[S1]** Ghosh, Schneider, Reinicke & Eickhoff, *A Survey on LLM-Assisted Clinical Trial Recruitment*,
  IJCNLP 2025 — the field map. Read first if you want orientation before depth.
  <https://arxiv.org/abs/2506.15301>

## Eligibility criteria as data
- **[D1]** Kury et al., *Chia, a large annotated corpus of clinical trial eligibility criteria*,
  Scientific Data 7:281 (2020) — 12,409 annotated criteria, 15 entity types, criteria as DAGs.
  <https://www.nature.com/articles/s41597-020-00620-0>
- **[D2]** *The Leaf Clinical Trials Corpus*, Scientific Data (2022) — a second annotated criteria resource.
  <https://www.nature.com/articles/s41597-022-01521-0>
- **[D3]** Yuan & Weng et al., *Criteria2Query: a natural language interface to clinical databases for
  cohort definition*, JAMIA 26(4):294–305 (2019) — the classical parse-to-query pipeline, with honest
  error rates per stage. See also Criteria2Query 3.0 (2024) for the LLM version.
  <https://academic.oup.com/jamia/article-abstract/26/4/294/5308980>

## Retrieval and ranking
- **[R1]** Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- **[M1]** Chen et al., *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text
  Embeddings Through Self-Knowledge Distillation*, Findings of ACL 2024 — BGE-M3; dense + sparse +
  multi-vector from one model, 8192-token context.
  <https://arxiv.org/abs/2402.03216>
- **[M2]** Jin et al., *MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs
  for zero-shot biomedical information retrieval*, Bioinformatics 39(11) (2023) — retriever and reranker
  trained jointly, which is why it is a natural Phase 7 candidate.
  <https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842>
- **[M3]** Wang & Sun, *Trial2Vec: Zero-Shot Clinical Trial Document Similarity Search using
  Self-Supervision*, Findings of EMNLP 2022 — trial-specific document representation.
- **[F1]** Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank
  Learning Methods*, SIGIR 2009 — the one-line fusion baseline that is hard to beat.
- **[K1]** Nogueira et al., *Document Ranking with a Pretrained Sequence-to-Sequence Model*, Findings of
  EMNLP 2020 — monoT5 cross-encoder reranking.
- **[K2]** Sun et al., *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking
  Agents*, EMNLP 2023 — RankGPT, listwise LLM reranking.
- **[Q1]** Gao, Ma, Lin & Callan, *Precise Zero-Shot Dense Retrieval without Relevance Labels*, ACL 2023 —
  HyDE. Generate a hypothetical trial from the patient profile, retrieve with its embedding.
  <https://arxiv.org/abs/2212.10496>

## Eligibility reasoning — closest prior work
- **[E1]** Jin, Wang, … Lu, *Matching patients to clinical trials with large language models*,
  Nature Communications 15:9074 (2024) — **TrialGPT**. Retrieval → criterion-level matching → ranking;
  87.3% criterion-level accuracy; 42.6% screening-time reduction. Read this one closely: it is the
  nearest neighbour to TrialMatch-RAG, and [research edge](research-edge.md) depends on what its release
  contains.
  <https://www.nature.com/articles/s41467-024-53081-z> · code: <https://github.com/ncbi-nlp/TrialGPT>
- **[E2]** Wornow et al., *Zero-Shot Clinical Trial Patient Matching with LLMs*, NEJM AI 2(1) (2025) —
  SOTA on n2c2 2018 with prompting alone; clinicians judged justifications coherent in 97% of correct
  decisions but only 75% of incorrect ones. That gap is the honest case for grounding checks.
  <https://arxiv.org/abs/2402.05125>
- **[E3]** Jullien, Bogatu, Unsworth & Freitas, *Controlled LLM-based Reasoning for Clinical Trial
  Retrieval* (2024) — set-guided reasoning, evaluated on TREC 2022: nDCG@10 0.693, P@10 0.73. Directly
  comparable numbers.
  <https://arxiv.org/abs/2409.18998>
- **[N1]** Stubbs, Filannino, Soysal, Henry & Uzuner, *Cohort selection for clinical trials: n2c2 2018
  shared task track 1*, JAMIA 26(11):1163–1171 (2019) — the second benchmark, patient-level over 13
  criteria. Requires a data-use agreement.
  <https://academic.oup.com/jamia/article-abstract/26/11/1163/5575392>

## Grounding and attribution
- **[G1]** Gao et al., *Enabling Large Language Models to Generate Text with Citations*, EMNLP 2023 —
  the ALCE benchmark; automatic citation precision/recall, adaptable to criterion-level grounding.

## Research edge
- **[E4]** *Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials
  Matching* (SatIR), COLM 2026 — SMT and relational algebra for criteria, LLMs only for translating
  ambiguous clinical text into explicit constraints. Reports 1.8–3.2× higher eligible-trial recall than
  TrialGPT-style retrieval on a TREC 2022 subset.
  <https://arxiv.org/abs/2604.08849>

---
[← specs index](README.md)
