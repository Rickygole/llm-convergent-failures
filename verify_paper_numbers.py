#!/usr/bin/env python3
"""Recompute every headline number in the ICDM 2026 paper from released predictions.

Usage:  python verify_paper_numbers.py [--json out.json]

Reads only predictions/, so any figure in the paper can be independently checked.
Prints PASS/FAIL against the values printed in the camera-ready.
"""
import json, math, argparse, os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
MID = ["llama", "qwen", "gemma3n"]

MEDQA = {"llama": "llama_results.json", "qwen": "qwen_results.json",
         "gemma3n": "gemma3n_results.json", "gpt4o": "medqa_gpt4o_results.json",
         "deepseek": "deepseek_results.json", "flan_t5": "flan_t5_results.json"}
MEDMCQA = {m: f"medmcqa_{m}_results.json" for m in
           ["llama", "qwen", "gemma3n", "gpt4o", "flan_t5"]}


def load(dataset, files):
    out = {}
    for m, f in files.items():
        p = os.path.join(ROOT, "predictions", dataset, f)
        if os.path.exists(p):
            out[m] = {r["idx"]: r for r in json.load(open(p))}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def ids_of(M, keys):
    return sorted(set.intersection(*[set(M[k]) for k in keys]))


def unanimous_wrong(M, keys):
    """(n_all_wrong, n_unanimous_wrong) over models `keys`."""
    allw = unan = 0
    for i in ids_of(M, keys):
        g = M[keys[0]][i]["gold"]
        preds = [M[k][i]["pred"] for k in keys]
        if all(p != g for p in preds):
            allw += 1
            if len(set(preds)) == 1:
                unan += 1
    return allw, unan


def detector(M, mid, frontier):
    """Returns dict of detector tier statistics."""
    keys = mid + [frontier]
    ids = ids_of(M, keys)
    hr = hrw = cf = cfw = cons_err = 0
    for i in ids:
        g = M[mid[0]][i]["gold"]
        votes = [M[m][i]["pred"] for m in mid]
        if len(set(votes)) != 1:
            continue
        c = votes[0]
        wrong = c != g
        cons_err += wrong
        if M[frontier][i]["pred"] != c:
            hr += 1; hrw += wrong
        else:
            cf += 1; cfw += wrong
    total_err = sum(1 for i in ids if M[mid[0]][i]["correct"] == 0)  # placeholder
    dataset_err = 0
    for i in ids:
        g = M[mid[0]][i]["gold"]
        votes = [M[m][i]["pred"] for m in mid]
        if Counter(votes).most_common(1)[0][0] != g:
            dataset_err += 1
    return dict(n=len(ids), hr=hr, hr_wrong=hrw, hr_prec=100 * hrw / hr,
                hr_ci=wilson(hrw, hr), route=100 * hr / len(ids),
                cf=cf, cf_wrong=cfw, cf_rate=100 * cfw / cf, cf_ci=wilson(cfw, cf),
                consensus_errors=cons_err,
                discriminative_recall=100 * hrw / cons_err,
                operational_recall=100 * hrw / dataset_err,
                dataset_majority_errors=dataset_err)


def check(results, label, got, want, tol=0.05):
    ok = abs(got - want) <= tol
    results.append(dict(check=label, computed=round(got, 2), paper=want, pass_=ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: computed {got:.2f}, paper {want}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    R = []

    Q = load("medqa", MEDQA)
    C = load("medmcqa", MEDMCQA)

    print("\n== Table I: unanimous wrong rates (conditional) ==")
    for label, M, keys, want in [
        ("MedQA 3 mid tier", Q, MID, 41.9),
        ("MedQA 4 incl. DeepSeek", Q, MID + ["deepseek"], 33.6),
        ("MedQA 4 incl. GPT-4o", Q, MID + ["gpt4o"], 26.0),
        ("MedQA 5 strong", Q, MID + ["deepseek", "gpt4o"], 24.6),
        ("MedMCQA 4 incl. GPT-4o", C, MID + ["gpt4o"], 23.6),
    ]:
        if not all(k in M for k in keys):
            print(f"  [SKIP] {label} (missing predictions)"); continue
        allw, unan = unanimous_wrong(M, keys)
        check(R, f"{label} unanimous-wrong %", 100 * unan / allw, want)

    print("\n== Section V-A: unconditional silent-failure rates ==")
    for label, M, keys, want in [
        ("MedQA 3 mid tier", Q, MID, 9.2),
        ("MedQA 4 incl. DeepSeek", Q, MID + ["deepseek"], 3.8),
        ("MedQA 5 strong", Q, MID + ["deepseek", "gpt4o"], 1.1),
        ("MedMCQA 4 incl. GPT-4o", C, MID + ["gpt4o"], 2.5),
    ]:
        if not all(k in M for k in keys):
            print(f"  [SKIP] {label}"); continue
        _, unan = unanimous_wrong(M, keys)
        check(R, f"{label} unconditional %", 100 * unan / len(ids_of(M, keys)), want)

    print("\n== Table IV: detector (frontier = GPT-4o) ==")
    dq, dc = detector(Q, MID, "gpt4o"), detector(C, MID, "gpt4o")
    check(R, "MedQA HIGH_RISK precision", dq["hr_prec"], 96.1)
    check(R, "MedQA HIGH_RISK route %", dq["route"], 8.0)
    check(R, "MedQA discriminative recall", dq["discriminative_recall"], 83.8)
    check(R, "MedQA operational recall", dq["operational_recall"], 18.6)
    check(R, "MedQA CONFIDENT wrong %", dq["cf_rate"], 4.5)
    check(R, "MedMCQA HIGH_RISK precision", dc["hr_prec"], 83.0)
    check(R, "MedMCQA HIGH_RISK route %", dc["route"], 6.7)
    check(R, "MedMCQA discriminative recall", dc["discriminative_recall"], 68.7)  # CORRECTED
    check(R, "MedMCQA operational recall", dc["operational_recall"], 13.0)
    check(R, "MedMCQA CONFIDENT wrong %", dc["cf_rate"], 8.2)
    print(f"  MedQA  CIs: HIGH_RISK {dq['hr_ci']}  CONFIDENT {dq['cf_ci']}")
    print(f"  MedMCQA CIs: HIGH_RISK {dc['hr_ci']}  CONFIDENT {dc['cf_ci']}")

    print("\n== Section VI-G: leave-one-out over mid-tier membership ==")
    for name, M, want in [("MedQA", Q, {"llama": 94.0, "qwen": 95.1, "gemma3n": 93.8}),
                          ("MedMCQA", C, {"llama": 86.2, "qwen": 85.3, "gemma3n": 81.6})]:
        for drop in MID:
            keep = [m for m in MID if m != drop]
            d = detector(M, keep, "gpt4o")
            check(R, f"{name} LOO drop-{drop} precision", d["hr_prec"], want[drop])

    print("\n== Section V-E: deployed-system accuracy (perfect review on HIGH_RISK) ==")
    ids = ids_of(Q, MID + ["gpt4o"])
    mid_ok = hr = hr_midok = 0
    for i in ids:
        g = Q["llama"][i]["gold"]
        votes = [Q[m][i]["pred"] for m in MID]
        maj = Counter(votes).most_common(1)[0][0]
        mid_ok += maj == g
        if len(set(votes)) == 1 and Q["gpt4o"][i]["pred"] != votes[0]:
            hr += 1; hr_midok += maj == g
    n = len(ids)
    check(R, "mid-tier ensemble accuracy", 100 * mid_ok / n, 58.5)
    check(R, "detector-routed accuracy", 100 * (mid_ok - hr_midok + hr) / n, 66.2)  # CORRECTED
    check(R, "GPT-4o frontier-only accuracy",
          100 * sum(Q["gpt4o"][i]["pred"] == Q["gpt4o"][i]["gold"] for i in ids) / n, 88.5)

    print("\n== Section V-F: vote-share 0.75 partition (MedQA) ==")
    tot = hr_n = oth = oth_wrong = 0
    for i in ids:
        g = Q["llama"][i]["gold"]
        votes = [Q[m][i]["pred"] for m in MID]
        allv = votes + [Q["gpt4o"][i]["pred"]]
        top, cnt = Counter(allv).most_common(1)[0]
        if cnt != 3:
            continue
        tot += 1
        if len(set(votes)) == 1 and Q["gpt4o"][i]["pred"] != votes[0]:
            hr_n += 1
        else:
            oth += 1; oth_wrong += top != g
    check(R, "vote-share 0.75 HIGH_RISK n", hr_n, 102, tol=0)
    check(R, "vote-share 0.75 other-3-of-4 n", oth, 326, tol=0)  # CORRECTED
    check(R, "other-3-of-4 wrong %", 100 * oth_wrong / oth, 12.9)

    print("\n== Section V-D: shuffle cohort definition ==")
    if "flan_t5" in C:
        _, u71 = unanimous_wrong(C, MID + ["gpt4o"])
        n48 = 0
        for i in ids_of(C, MID + ["gpt4o", "flan_t5"]):
            g = C["llama"][i]["gold"]
            v = [C[m][i]["pred"] for m in MID + ["gpt4o"]]
            if all(p != g for p in v) and len(set(v)) == 1 and C["flan_t5"][i]["pred"] != g:
                n48 += 1
        check(R, "MedMCQA 4-strong unanimous-wrong n", u71, 71, tol=0)
        check(R, "  ... and Flan also wrong (shuffle cohort) n", n48, 48, tol=0)

    npass = sum(r["pass_"] for r in R)
    print(f"\n{'='*58}\n{npass}/{len(R)} checks PASS")
    if args.json:
        json.dump(R, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")
    return 0 if npass == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
