"""Write comparison md from graduated boost eval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "06_AGGREGATOR/results/graduated_boost_eval/claude_comparison_graduated.csv")
tau = 0.32
scored = df[df["vulnera_pct"].notna()]
vuln = scored[scored["claude_vuln"]]
safe = scored[~scored["claude_vuln"]]

lines = [
    "# VULNERA vs claude.md — graduated signature boost",
    "",
    "Policy: corroboration (omega=0.15) + graduated blend toward plateau (no hard 84.6% jump).",
    "",
    "## Summary vs hard plateau",
    "",
    "| Metric | Hard plateau | Graduated |",
    "|--------|-------------:|----------:|",
    f"| Triage agree | 67.6% | {scored['vuln_match'].mean():.1%} |",
    f"| Vuln recall (flagged) | 68.8% | {vuln['vulnera_flagged'].mean():.1%} |",
    f"| Safe specificity | 65.2% | {(~safe['vulnera_flagged']).mean():.1%} |",
    f"| Mean |Δ| vs claude | 29.4 | {scored['delta'].abs().mean():.1f} |",
    "",
    "## PrimeVul F1",
    "",
    "| Split | ML-only | Plateau+corr | Graduated |",
    "|-------|--------:|-------------:|----------:|",
    "| valid | 0.4811 | 0.4914 | **0.4914** |",
    "| test | 0.4697 | 0.4768 | **0.4768** |",
    "",
    "## Per function",
    "",
    "| File | Function | Claude | VULNERA | Boost | Match |",
    "|------|----------|-------:|--------:|-------|-------|",
]
for _, row in df.iterrows():
    match = "✓" if row.get("vuln_match") else "✗"
    vp = f"{row['vulnera_pct']:.1f}%" if pd.notna(row["vulnera_pct"]) else "—"
    mode = row.get("boost_mode") or "—"
    lines.append(
        f"| {row['file']} | `{row['function']}` | {int(row['claude_pct'])}% | {vp} | {mode} | {match} |"
    )

(ROOT / "tests/claude_vs_vulnera_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote comparison md")
