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
- 95.25% of the 28,787 instances mapped to a filing (95.39% after exact dedup). 93.09% of all instances
  are reproduced as exact strings by a port of your `parse_xbrl_report.ipynb` run on the downloaded
  filings; the rest come from a fuzzy fallback. On the exact-string matches, the port's labels equal the
  gold label set for 25,768 of 25,770 instances, and every gold value occurs in the assigned filing.
  (My earlier token-level matcher, without your serializer, reached 94.29%.)

Four observations that may interest you:

1. 10 companies appear to contribute two fiscal years (FY2023 and FY2024 10-Ks). Several contexts occur
   only in one year's filing, for example year-specific portfolio counts.
2. About 15% of multi-token table cells in the contexts are exact duplicates ("Date Declared Date
   Declared"). This is consistent with `extract_table` joining the text of every nested `p`/`div`/`span`
   inside a cell. Mentioned in case it is unintended.
3. In `extract_text`, a `div` whose only content is a `p` yields the same text twice. The `div` is emitted
   as a **negative**, because its facts are not direct children, and the `p` as a **positive**. Negatives
   are not added to `seen_texts`, so both survive. In the filings I mapped, 4,746 evaluation contexts have
   more than one label set in their own filing. This may explain the duplicate contexts with differing
   answers in FinNI-eval-v2.
4. The repository's `us-gaap-2024.xsd` lacks some elements that evaluation filings use (for example
   `StockRepurchaseProgramAuthorizedAmount1`). The union of the 2023 and 2024 us-gaap element lists has
   exactly the 17,688 concepts the paper mentions. Is that what `us-gaap-2024_taxonomy.json` contains?

The method and the CSVs are public in my repository: <link to be added by the owner>.

Thank you!
