# project2-xbrl-frontier

Review and PRD for the second portfolio project (fine-tuned small-model XBRL fact extraction measured as an accuracy / cost / latency frontier), produced 2026-10-01.

- `REVIEW.md`: Part 1, the critical review with the verdict (APPROVE WITH CHANGES).
- `PRD.md`: Part 2, the PRD with the changes applied; v2 re-planned for a $10 API-only budget, no rented GPUs, Mac M1 Pro plus RTX 3050 plus free Kaggle T4, about 90 hours, FinNI only.
- `config/thresholds.yaml`: every numeric gate, pin, seed and budget; the PRD names keys, never values.
- `config/prices_2026-10-01.yaml`: dated API and GPU price sheet with sources.
- `research/evidence_brief.md`: condensed, verified facts behind the review.
- `research/dossier/*.txt`: the full per-topic research reports with URLs and quotes (papers, data releases, contamination practice, MLX, GPU rental, API pricing, training-data scale, and ten gap follow-ups).
- `research/recovered_eval_manifest.json`: 111 of the 142 FinTagging evaluation registrants recovered via EDGAR full-text search; `fts_eval*.json`, `fts_training.json`, `overlap_classification.json`, `edgar_submissions_train10.json` are the raw query results behind it.
- `research/bm25.jsonl`: the 17,388-concept 2024 US-GAAP index shipped with FinTagging.
