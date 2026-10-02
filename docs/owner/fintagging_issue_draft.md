# Draft GitHub issue for The-FinAI/FinTagging (owner posts; not posted by the builder)

**Title:** Request: CIK/accession list for the 142 evaluation filings, and the FinNI scoring script

Hi, and thanks for releasing FinTagging and the FinNI/FinCL evaluation sets.

I am building an open, reproducible accuracy / cost / latency comparison on `TheFinAI/FinNI-eval-v2`
(revision `e3e0aba23b5c3df17a71c8c013030182d013a538`, 28,787 rows). Two things would make the results
directly comparable to yours.

**1. The filing list behind the evaluation set.** The paper describes 142 10-K filings (filed 2023-02-13 to
2025-02-13), but the release carries no CIK, accession number or company per instance. I need the mapping
to resample over filings for confidence intervals, to slice by company and fiscal year, and to exclude
those registrants from training data (contamination control). Could you share the 142 (CIK, accession
number) pairs, or the per-instance source filing if you have it?

**2. The FinNI scorer for LLM outputs.** Appendix E defines a match as entity string AND type equal, with
set-based P/R/F1. I could not find the script that produced the published FinNI numbers. In particular I
would like to know: set vs multiset counting, global vs per-instance aggregation, value normalization beyond
the prompt's rules, and how unparseable outputs were handled. Could you publish or share it?

**What I can offer in return.** I recovered the manifest independently and am happy to contribute it back:

- 129 of the 142 registrants (139 filings) recovered by exact-phrase EDGAR full-text search (11-word
  phrases from evaluation contexts, forms=10-K, the benchmark's filing window, only queries resolving to a
  single registrant), then confirmed by matching every evaluation context against the downloaded filings.
- 94.3% of the 28,787 instances mapped to a filing. For exact matches, every gold value occurs in the
  assigned filing.
- One observation that may interest you: 10 companies appear to contribute two fiscal years (FY2023 and
  FY2024 10-Ks). Several contexts occur only in one year's filing, for example year-specific
  portfolio counts.
- About 15% of multi-token table cells in the contexts are exact duplicates ("Date Declared Date
  Declared"). This is consistent with `parse_xbrl_report.ipynb` joining the text of nested `p`/`div`/`span`
  elements inside a cell. Mentioned in case it is unintended.

The method and the CSVs are public in my repository: <link to be added by the owner>.

Thank you!
