"""Self-contained HTML summary for a completed benchmark run."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from .models import RunPlan


def _s(value: Any) -> str:
    return escape(str(value))


def render_report(plan: RunPlan, results: list[dict[str, Any]], run_id: str) -> str:
    completed = sorted((r for r in results if r["status"] == "completed"), key=lambda r: r["gini"], reverse=True)
    rows = []
    for rank, result in enumerate(completed, 1):
        license_text = _s(result.get("license") or "Unknown")
        if result.get("attribution_required"):
            license_text += " · attribution required"
        rows.append(
            f"<tr><td>{rank}</td><td><strong>{_s(result['name'])}</strong></td>"
            f"<td class='num'>{result['gini']:.4f}</td><td class='num'>{result['auc']:.4f}</td>"
            f"<td class='num'>{result['context_rows']:,}</td><td class='num'>{result['feature_count']:,}</td>"
            f"<td class='num'>{result['scored_rows']:,}</td><td class='num'>{result['seconds']:,.1f}s</td>"
            f"<td>{license_text}</td>"
            f"<td>{_s('; '.join(result.get('adaptations', [])) or 'Full reference data')}</td></tr>"
        )
    other = []
    for result in results:
        if result["status"] != "completed":
            other.append(f"<tr><td>{_s(result['name'])}</td><td><span class='badge'>{_s(result['status'])}</span></td>"
                         f"<td>{_s(result.get('reason') or result.get('error') or '')}</td></tr>")
    config_rows = []
    planned = {model.name: model for model in plan.models}
    for result in completed:
        item = planned[result["name"]]
        checkpoint = result.get("checkpoint", {})
        checkpoint_text = checkpoint.get("sha256") or checkpoint.get("reported_reference") or checkpoint.get("resolution", "unavailable")
        config_rows.append(
            f"<tr><td>{_s(result['name'])}</td><td><code>{_s(json.dumps(result.get('tuning_params', item.tuning_params), default=str))}</code></td>"
            f"<td><code>{_s(json.dumps(result.get('model_params', item.model_params), default=str))}</code></td>"
            f"<td>{_s(result.get('weights') or item.weights or '—')}<br><small>{_s(checkpoint_text)}</small></td></tr>"
        )
    best = f"{completed[0]['gini']:.4f}" if completed else "—"
    gpu = plan.profile.get("gpu", {})
    hardware = f"{_s(plan.profile.get('cpu_count'))} CPUs · {plan.profile['ram_total_bytes']/1024**3:.1f} GiB RAM"
    if gpu.get("available"):
        hardware += f" · {_s(gpu.get('name'))} ({gpu.get('vram_bytes', 0)/1024**3:.1f} GiB VRAM)"
    split_text = "Explicit train/validation column" if plan.split_source == "explicit" else f"Last three calendar months, cutoff {_s(plan.cutoff)}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TabTune baseline · {_s(run_id)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#17233a; background:#f3f6fb; }}
* {{ box-sizing:border-box }} body {{ margin:0; }} main {{ max-width:1240px; margin:40px auto; padding:0 24px 60px; }}
header {{ background:linear-gradient(120deg,#142a56,#395fba); color:white; border-radius:22px; padding:32px 38px; box-shadow:0 14px 35px #1a326329; }}
h1 {{ margin:4px 0 8px; font-size:2rem }} h2 {{ margin:0 0 18px; font-size:1.2rem }} p {{ line-height:1.5 }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; opacity:.8 }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:20px 0; }}
.card, section {{ background:white; border:1px solid #e1e7f1; border-radius:16px; box-shadow:0 5px 18px #1a32630c; }}
.card {{ padding:18px 20px }} .card .label {{ color:#697991; font-size:.8rem }} .card .value {{ font-size:1.55rem; font-weight:750; margin-top:6px }}
section {{ padding:24px; margin-top:18px; overflow:auto }} table {{ width:100%; border-collapse:collapse; font-size:.9rem }}
th {{ text-align:left; color:#566782; font-weight:650; background:#f6f8fc }} td,th {{ padding:11px 12px; border-bottom:1px solid #e8edf5; vertical-align:top }}
tr:last-child td {{ border-bottom:0 }} .num {{ font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#eef1f7; color:#465978; font-size:.78rem }}
.muted {{ color:#697991 }} code {{ background:#f0f3f8; padding:2px 5px; border-radius:4px; overflow-wrap:anywhere }}
</style></head><body><main>
<header><div class="eyebrow">Zero-shot binary classification</div><h1>TabTune baseline</h1>
<div>Run {_s(run_id)} · Gini = 2 × ROC AUC − 1 · Positive class: {_s(plan.positive_label)}</div></header>
<div class="cards"><div class="card"><div class="label">Best Gini</div><div class="value">{best}</div></div>
<div class="card"><div class="label">Models scored</div><div class="value">{len(completed)}</div></div>
<div class="card"><div class="label">Reference rows</div><div class="value">{plan.train_rows:,}</div></div>
<div class="card"><div class="label">Validation rows</div><div class="value">{plan.validation_rows:,}</div></div></div>
<section><h2>Leaderboard</h2><p class="muted">Each successful model is evaluated on the same complete validation set. Reference rows and selected features can differ by model; adaptations are shown below.</p>
<table><thead><tr><th>#</th><th>Model</th><th class="num">Gini</th><th class="num">ROC AUC</th><th class="num">Context</th><th class="num">Features</th><th class="num">Scored</th><th class="num">Time</th><th>Weights license</th><th>Adaptations</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="10">No model completed successfully.</td></tr>'}</tbody></table></section>
<section><h2>Dataset and environment</h2><p><strong>Split:</strong> {split_text}<br>
<strong>Features supplied:</strong> {len(plan.feature_cols):,} · <strong>Missing cells:</strong> {plan.profile['missing_cells']:,}<br>
<strong>Reference labels:</strong> {_s(plan.profile.get('reference_target_counts'))} · <strong>Validation labels:</strong> {_s(plan.profile.get('validation_target_counts'))}<br>
<strong>Hardware:</strong> {hardware}<br><strong>TabTune:</strong> {_s(plan.profile.get('tabtune_version'))} · <strong>Policy:</strong> {_s(plan.policy_version)}</p></section>
<section><h2>Parameters and checkpoints</h2><table><thead><tr><th>Model</th><th>Tuning parameters</th><th>Model parameters</th><th>Checkpoint source and identity</th></tr></thead>
<tbody>{''.join(config_rows) if config_rows else '<tr><td colspan="4">No completed models.</td></tr>'}</tbody></table></section>
<section><h2>Excluded and unsuccessful models</h2><table><thead><tr><th>Model</th><th>Status</th><th>Reason</th></tr></thead>
<tbody>{''.join(other) if other else '<tr><td colspan="3">None</td></tr>'}</tbody></table></section>
<section><h2>Run files</h2><p class="muted">See <code>results.json</code> for metrics and attempts, <code>manifest.json</code> for configuration and model plans, <code>scores/</code> for row-level validation probabilities, and <code>events.jsonl</code> for structured run logs.</p></section>
</main></body></html>"""
