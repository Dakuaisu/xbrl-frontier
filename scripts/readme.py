"""Generate the root README.md skeleton (PRD sections 1, 8, 11; T-032).

Every result number is "__" until a measured result file provides it; gates are named by key with their status
read from data/gates/summary.json when it exists ("not measured" otherwise); disclosure lines and the
contamination statement are read from recorded files at generation time (scripts/contamination_statement.py).
No value is typed by hand. The Mac pilot's format-check F1 never appears (O-021). Regenerate after each result.
"""
import json

from scripts.contamination_statement import build as contamination_statement
from xbrl_frontier import config

D = config.REPO_ROOT / "data"
OUT = config.REPO_ROOT / "README.md"
B = "__"


def _j(p):
    return json.loads(p.read_text()) if p.exists() else None


def build() -> str:
    ser = _j(D / "manifest_work" / "serializer_rebuild.json")
    cor = _j(D / "corpus_report.json")
    pool, final = _j(D / "contamination_audit" / "summary_pool.json"), _j(D / "contamination_audit" / "summary_final.json")
    gates = _j(D / "gates" / "summary.json")
    import csv
    man = list(csv.DictReader(open(config.docs_path(config.get("pins.eval_manifest")))))
    companies = len({m["cik"] for m in man})
    L = ["# xbrl-frontier", "",
         "An accuracy / cost-per-filing / latency frontier for fine-tuned Qwen3-0.6B and Qwen3-1.7B students on FinTagging "
         "FinNI, measured against pinned frontier APIs on identical instances, with fp16 / INT8 / INT4 serving on a "
         "market-priced T4 and on owned hardware. The deliverable is the harness and the chart, not a model. Spec: "
         "`docs/PRD.md`; decisions: `docs/TRADEOFFS.md`; findings: `docs/OPEN.md`.", "",
         "## Headline", "",
         f"{B}% of {' / '.join(config.get('frontier.running_arms'))}'s FinNI F1 at {B}x lower cost per filing and {B} ms p95 at {B} requests per second on one T4 "
         "(paired company-level bootstrap over the shared frontier sample; blanks fill only from measured runs).", "",
         "## Gates", "", "| gate (key under `gates`) | status | point | interval |", "|---|---|---|---|"]
    rows = gates["gates"] if gates else [{"gate": k, "status": "not measured", "point": None, "ci": None}
                                         for k in config.get("gates") if k not in ("cost_comparator_rule", "cost_gate_regime",
                                                                                  "cost_regimes", "decision_rule", "headline_margin_f1",
                                                                                  "fresh_slice_control_gap_reported")]
    for r in rows:
        pt = B if r.get("point") is None else f"{r['point']:.4f}"
        ci = B if not r.get("ci") else f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        L.append(f"| `{r['gate']}` | {r['status']} | {pt} | {ci} |")
    L += ["", "Decisions follow `gates.decision_rule`: an interval that straddles the threshold is reported as inconclusive, "
          "and an inconclusive headline gate ships without the claim.", "",
          "## Results", "",
          f"- Pareto, F1 vs $ per filing (offline and online regimes): {B} (`data/charts/pareto_*.vl.json` when measured)",
          f"- Learning curve, F1 vs distinct training instances, overall and table vs text: {B} (`data/charts/learning_curve.vl.json`)",
          f"- Serving latency and throughput, T4 and RTX 3050: {B} (`data/charts/latency.vl.json`); the 3050 rows carry "
          f"\"{config.get('serving.hardware.rtx3050_wsl2.cost_label')}\"",
          f"- Encoder baseline (SEC-BERT): {B}; zero-shot base models: {B}; fresh-slice gap and control gap: {B}", "",
          "## Disclosures (from recorded files)", ""]
    planned, running = config.get("pins.frontier_arms"), config.get("frontier.running_arms")
    L.append(f"- Frontier arms run: {', '.join(f'`{a}`' for a in running)}. Planned but not run: "
             f"{', '.join(f'`{a}`' for a in planned if a not in running)} (owner decision T-033: no key and $0 billed spend; "
             "this goes beyond the PRD cut order, and the cause is money, not results).")
    L.append("- Frontier collection runs on the Gemini free tier: tokens, the returned model field and latency come from the free "
             "tier; free-tier latency may include throttling; dollars per filing are measured tokens priced at the dated sheet's "
             "batch (accuracy) and sync (latency) rates, with $0 billed. Google may use free-tier inputs; they are public SEC text (T-033).")
    L.append(f"- Evaluation filings recovered: {companies} of {config.get('contamination.eval_manifest_required_recovered')} registrants "
             f"({len(man)} filings); `contamination.eval_manifest_required_recovered` not met (O-010).")
    if ser:
        L.append(f"- Instances mapped to a filing: {ser['assignment_rate_raw']:.2%} raw / {ser['assignment_rate_dedup']:.2%} dedup with the "
                 f"benchmark's own serializer; Phase 0 token matcher {ser['phase0_token_matcher']['assignment_rate_raw']:.2%} / "
                 f"{ser['phase0_token_matcher']['assignment_rate_dedup']:.2%} (O-011).")
    if cor:
        t, d = cor["splits"]["train"], cor["splits"]["dev"]
        L.append(f"- Training corpus: {t['used']:,} (cap {t['cap']:,} not reached after decontamination, O-016); dev slice "
                 f"{d['used']:,} (cap {d['cap']:,} not reached after decontamination, O-016). Curve points count distinct instances (T-016).")
    if pool and final:
        L.append(f"- Contamination audit: {pool['flagged']:,} of {pool['audited']:,} eligible training instances flagged and removed; "
                 f"{final['flagged']} flagged in the final corpus of {final['audited']:,}.")
    L += ["", contamination_statement(), "",
          "## Resume bullets (blanks fill only from measured runs; PRD section 11)", "",
          f"- Fine-tuned a {B}B-parameter open model on {B} contamination-audited iXBRL instances that reached {B}% of {B}'s FinNI F1 "
          f"on the FinTagging benchmark (paired bootstrap over {B} filings) and served it with vLLM at {B} ms p95 and {B} requests/s "
          f"on a single T4, {B}x cheaper per filing at market rate; total API spend ${B}.",
          f"- Published the F1-versus-training-size curve for regulatory table extraction: {B}% of final F1 at {B} instances, "
          f"with tables lagging text by {B} points at every size.",
          f"- Recovered FinTagging's unpublished evaluation manifest from EDGAR full-text search, mapped {B}% of evaluation instances "
          f"to their filings, and shipped the exclusion audit ({B} flagged, {B} remaining) plus a post-cutoff fresh slice with a {B}-point gap.",
          f"- Measured fp16 / INT8 / INT4 serving variants on Turing and Ampere consumer hardware with per-fact flip rates ({B}% at INT4) "
          "rather than aggregate F1 alone; every number reproducible from pinned revisions and a dated price sheet.", "",
          "## Reproducibility", "",
          f"- FinNI-eval-v2 `{config.get('pins.eval_finni.hf_revision')}`; students "
          + ", ".join(f"`{s['id']}@{s['hf_revision'][:12]}`" for s in config.get("pins.students"))
          + f"; encoder `{config.get('pins.encoder_baseline.id')}@{config.get('pins.encoder_baseline.hf_revision')[:12]}`.",
          f"- Prices: `docs/{config.get('pins.price_sheet')}` (observed {config.prices()['observed']}); every dollar figure cites it.",
          "- All gates, pins, seeds and budgets: `docs/config/thresholds.yaml`. Owner-run steps: `docs/owner/RUNBOOK.md`.", ""]
    return "\n".join(L)


def main() -> None:
    OUT.write_text(build())
    print(OUT.read_text())


if __name__ == "__main__":
    main()
