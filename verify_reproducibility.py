"""
verify_reproducibility.py

End-to-end reproducibility check for the paper:
"When LLM Ensembles Agree They Are Wrong: Convergent Failures in
Clinical Question Answering"

Runs 17 checks against the released JSON files and prints PASS/FAIL.
Expected output is 17/17 PASS.

Usage:
    pip install -r requirements.txt
    python verify_reproducibility.py [path_to_icdm_release]

If no path is given, defaults to the script's parent directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, ttest_ind
from sklearn.metrics import cohen_kappa_score


def load_idx(release: Path, name: str) -> dict:
    """Load a prediction JSON and key it by question idx."""
    return {r['idx']: r for r in json.loads((release / name).read_text())}


def main(release: Path) -> int:
    if not release.exists():
        print(f"ERROR: release directory not found: {release}")
        return 1

    print(f"Running reproducibility checks against: {release}\n")
    results: list[tuple[str, bool, str]] = []

    # --- 1. Per-model accuracies ----------------------------------------
    acc_targets = {
        'predictions/medqa/llama_results.json':       0.520,
        'predictions/medqa/qwen_results.json':        0.595,
        'predictions/medqa/gemma3n_results.json':     0.539,
        'predictions/medqa/deepseek_results.json':    0.777,
        'predictions/medqa/medqa_gpt4o_results.json': 0.885,
        'predictions/medmcqa/medmcqa_llama_results.json': 0.5458,
        'predictions/medmcqa/medmcqa_gpt4o_results.json': 0.7720,
    }
    for fname, target in acc_targets.items():
        recs = json.loads((release / fname).read_text())
        acc = sum(r['correct'] == 1 for r in recs) / len(recs)
        results.append((f"Accuracy {fname.split('/')[-1]}",
                       abs(acc - target) < 0.005,
                       f"{acc:.4f} vs {target}"))

    # --- 2. 5-LLM MedQA unanimous-wrong --------------------------------
    strong = ['llama_results', 'qwen_results', 'gemma3n_results',
              'deepseek_results', 'medqa_gpt4o_results']
    ms = [load_idx(release, f'predictions/medqa/{f}.json') for f in strong]
    common = set.intersection(*[set(m.keys()) for m in ms])
    common = {i for i in common if all(m[i]['pred'] is not None for m in ms)}
    all_wrong = [i for i in common if all(m[i]['correct'] == 0 for m in ms)]
    unanim = sum(1 for i in all_wrong if len({m[i]['pred'] for m in ms}) == 1)
    results.append(("5-LLM MedQA unanimous wrong",
                   unanim == 14 and len(all_wrong) == 57,
                   f"{unanim}/{len(all_wrong)}"))

    # --- 3. Detector HIGH_RISK on MedQA --------------------------------
    mids = [load_idx(release, f'predictions/medqa/{f}.json')
            for f in ['llama_results', 'qwen_results', 'gemma3n_results']]
    frontier = load_idx(release, 'predictions/medqa/medqa_gpt4o_results.json')
    common = set.intersection(*[set(m.keys()) for m in mids]) & set(frontier.keys())
    common = {i for i in common
              if all(m[i]['pred'] is not None for m in mids)
              and frontier[i]['pred'] is not None}
    hr = hr_wrong = 0
    for i in common:
        if len({m[i]['pred'] for m in mids}) == 1:
            top = mids[0][i]['pred']
            if frontier[i]['pred'] != top:
                hr += 1
                if top != mids[0][i]['gold']:
                    hr_wrong += 1
    prec = hr_wrong / hr if hr else 0
    results.append(("Detector HIGH_RISK MedQA",
                   hr == 102 and abs(prec - 0.961) < 0.005,
                   f"n={hr}, precision={prec * 100:.1f}%"))

    # --- 4. Fisher exact 5-wrong baseline ------------------------------
    d = json.loads((release / 'analysis/audit_5wrong_baseline.json').read_text())
    for ds, target_p in [('medqa', 0.137), ('medmcqa', 0.221)]:
        t = d[ds]['trap']
        a = d[ds]['all_5_wrong']
        _, p = fisher_exact([[t['unanim'], t['n'] - t['unanim']],
                             [a['unanim'], a['n'] - a['unanim']]])
        results.append((f"Fisher exact {ds}",
                       abs(p - target_p) < 0.005,
                       f"p={p:.4f} vs {target_p}"))

    # --- 5. Programmatic PC trap vs nontrap (MedQA) --------------------
    records = json.loads((release / 'bias_labels/programmatic_pc_scores.json').read_text())
    trap = np.array([r['pc_score'] for r in records
                     if r['dataset'] == 'medqa' and r['subset'] == 'trap'])
    nontrap = np.array([r['pc_score'] for r in records
                        if r['dataset'] == 'medqa' and r['subset'] == 'nontrap'])
    t_stat, p_val = ttest_ind(trap, nontrap, equal_var=False)
    results.append(("Programmatic PC trap vs nontrap",
                   abs(t_stat + 2.18) < 0.05 and abs(p_val - 0.030) < 0.005,
                   f"t={t_stat:.3f}, p={p_val:.4f}"))

    # --- 6. Cohen's kappa for bias triangulation -----------------------
    mq = json.loads((release / 'bias_labels/medqa_shared_failure_bias_gpt4omini.json').read_text())
    mm = json.loads((release / 'bias_labels/medmcqa_bias_labels_gpt4omini.json').read_text())
    k_q = cohen_kappa_score([r['bias_type_llama70b'] for r in mq],
                            [r['bias_type_gpt']      for r in mq])
    k_m = cohen_kappa_score([r['bias_type_llama70b'] for r in mm],
                            [r['bias_type_gpt']      for r in mm])
    results.append(("Cohen's kappa MedQA",   abs(k_q - 0.126) < 0.005, f"{k_q:.3f}"))
    results.append(("Cohen's kappa MedMCQA", abs(k_m - 0.215) < 0.005, f"{k_m:.3f}"))

    # --- 7. Frontier-only baseline vs detector-routed (MedQA) ----------
    from collections import Counter
    valid = lambda p: isinstance(p, str) and p in 'ABCD'
    mids = [load_idx(release, f'predictions/medqa/{f}.json')
            for f in ['llama_results', 'qwen_results', 'gemma3n_results']]
    frontier = load_idx(release, 'predictions/medqa/medqa_gpt4o_results.json')
    common = set.intersection(*[set(m) for m in mids]) & set(frontier)
    common = {i for i in common
              if all(valid(m[i].get('pred')) for m in mids) and valid(frontier[i].get('pred'))}
    n = len(common)
    gpt = sum(1 for i in common if frontier[i]['pred'] == frontier[i]['gold'])
    midmaj = sum(1 for i in common
                 if Counter(m[i]['pred'] for m in mids).most_common(1)[0][0] == mids[0][i]['gold'])
    routed = 0
    for i in common:
        mp = [m[i]['pred'] for m in mids]
        g = mids[0][i]['gold']
        fp = frontier[i]['pred']
        if len(set(mp)) == 1:
            routed += 1 if fp != mp[0] else (mp[0] == g)
        else:
            routed += (Counter(mp).most_common(1)[0][0] == g)
    results.append(("Frontier-only MedQA accuracy",
                   abs(gpt / n - 0.885) < 0.005, f"{gpt / n * 100:.1f}%"))
    results.append(("Detector-routed beats mid-tier (+7.7pp)",
                   abs((routed - midmaj) / n * 100 - 7.7) < 0.3,
                   f"+{(routed - midmaj) / n * 100:.1f}pp"))

    # --- 8. MedMCQA Yule Q (guards corrected diversity file) -----------
    div = json.loads((release / 'analysis/exp3_diversity_measures.json').read_text())
    ss = [p['Q'] for p in div['medmcqa_pairs'] if p['kind'] == 'strong-strong']
    results.append(("MedMCQA strong-strong Q mean",
                   abs(sum(ss) / len(ss) - 0.633) < 0.005, f"{sum(ss) / len(ss):.3f}"))

    # --- Print results --------------------------------------------------
    print(f"{'CHECK':<40s} {'STATUS':<6s} VALUE")
    print("-" * 80)
    for name, ok, val in results:
        print(f"{name:<40s} {'PASS' if ok else 'FAIL':<6s} {val}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve()
    else:
        path = Path(__file__).parent.resolve()
    sys.exit(main(path))
