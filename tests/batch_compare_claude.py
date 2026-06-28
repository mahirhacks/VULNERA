"""Batch-scan 10_TEST corpus and compare calibrated scores to tests/claude.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "09_WEB" / "back_end"
TEST_DIR = ROOT / "10_TEST"
CLAUDE_MD = ROOT / "tests" / "claude.md"
THRESHOLD = 0.32

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def parse_claude_md(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "File" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        file_col, func_col, risk_col = parts[0], parts[1], parts[2]
        file_col = file_col.strip()
        func_col = func_col.strip("` ")
        m = re.search(r"(\d+)\s*%", risk_col)
        if not m:
            continue
        rows.append(
            {
                "file": file_col,
                "function": func_col,
                "claude_pct": int(m.group(1)),
                "claude_vuln": int(m.group(1)) >= 50,
            }
        )
    return rows


def run_batch() -> list[dict]:
    from pipeline.scan_pipeline import run_scan

    results: list[dict] = []
    files = sorted(TEST_DIR.glob("test_*.c"))
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name} ...", flush=True)
        try:
            payload = run_scan(
                source=path.read_text(encoding="utf-8"),
                filename=path.name,
                llm_provider="mock",
                max_functions=50,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            results.append({"file": path.name, "error": str(exc)})
            continue

        for fn in payload.functions:
            score = float(fn.get("function_score_calibrated") or 0.0)
            flagged = bool(fn.get("function_flagged"))
            results.append(
                {
                    "file": path.name,
                    "function": str(fn.get("name") or ""),
                    "vulnera_pct": round(score * 100, 1),
                    "vulnera_flagged": flagged,
                    "agreement_status": str(fn.get("agreement_status", "")),
                    "signature_boosted": bool(fn.get("signature_risk_boosted")),
                    "pattern_category": (fn.get("pattern_attribution") or {}).get("category"),
                }
            )
    return results


def compare(expected: list[dict], actual: list[dict]) -> list[dict]:
    by_key = {(r["file"], r["function"]): r for r in actual if "error" not in r}
    rows: list[dict] = []
    for exp in expected:
        key = (exp["file"], exp["function"])
        got = by_key.get(key)
        if got is None:
            rows.append({**exp, "vulnera_pct": None, "match": "missing", "delta": None})
            continue
        vuln_pct = got["vulnera_pct"]
        delta = vuln_pct - exp["claude_pct"]
        vuln_match = (vuln_pct >= THRESHOLD * 100) == exp["claude_vuln"]
        direction = "ok"
        if exp["claude_vuln"] and vuln_pct < 50:
            direction = "under"
        elif not exp["claude_vuln"] and vuln_pct >= 50:
            direction = "over"
        rows.append(
            {
                **exp,
                **got,
                "delta": round(delta, 1),
                "vuln_match": vuln_match,
                "direction": direction,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("vulnera_pct") is not None]
    missing = [r for r in rows if r.get("match") == "missing"]
    vuln_rows = [r for r in scored if r["claude_vuln"]]
    safe_rows = [r for r in scored if not r["claude_vuln"]]
    return {
        "total_expected": len(rows),
        "scored": len(scored),
        "missing": len(missing),
        "vuln_triage_accuracy": sum(1 for r in scored if r.get("vuln_match")) / len(scored) if scored else 0,
        "vuln_recall_flagged": sum(1 for r in vuln_rows if r.get("vulnera_flagged")) / len(vuln_rows) if vuln_rows else 0,
        "safe_specificity": sum(1 for r in safe_rows if not r.get("vulnera_flagged")) / len(safe_rows) if safe_rows else 0,
        "mean_abs_delta": sum(abs(r["delta"]) for r in scored if r.get("delta") is not None) / len(scored) if scored else 0,
        "under_vuln": sum(1 for r in vuln_rows if r.get("direction") == "under"),
        "over_safe": sum(1 for r in safe_rows if r.get("direction") == "over"),
    }


def main() -> None:
    expected = parse_claude_md(CLAUDE_MD)
    actual = run_batch()
    rows = compare(expected, actual)
    summary = summarize(rows)

    out = ROOT / "tests" / "claude_vs_vulnera_comparison.md"
    lines = [
        "# VULNERA vs claude.md — 10_TEST batch scan",
        "",
        f"Deployment threshold: **{THRESHOLD:.0%}** (function_flagged)",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Functions expected (claude.md) | {summary['total_expected']} |",
        f"| Functions scored | {summary['scored']} |",
        f"| Missing from scan | {summary['missing']} |",
        f"| Triage agree (vuln/safe @ τ) | {summary['vuln_triage_accuracy']:.1%} |",
        f"| Vuln recall (flagged) | {summary['vuln_recall_flagged']:.1%} |",
        f"| Safe specificity (not flagged) | {summary['safe_specificity']:.1%} |",
        f"| Mean |Δ| vs claude % | {summary['mean_abs_delta']:.1f} |",
        f"| Vuln scored &lt;50% (under) | {summary['under_vuln']} |",
        f"| Safe scored ≥50% (over) | {summary['over_safe']} |",
        "",
        "## Per function",
        "",
        "| File | Function | Claude | VULNERA | Δ | Flagged | Pattern | Match |",
        "|------|----------|-------:|--------:|--:|:-------:|---------|-------|",
    ]
    for r in rows:
        if r.get("match") == "missing":
            lines.append(f"| {r['file']} | `{r['function']}` | {r['claude_pct']}% | — | — | — | — | **MISSING** |")
            continue
        match = "✓" if r.get("vuln_match") else "✗"
        pat = r.get("pattern_category") or "—"
        lines.append(
            f"| {r['file']} | `{r['function']}` | {r['claude_pct']}% | {r['vulnera_pct']:.1f}% | "
            f"{r['delta']:+.1f} | {r['vulnera_flagged']} | {pat} | {match} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
