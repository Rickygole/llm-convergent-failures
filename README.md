# Convergent failures in clinical LLM ensembles: reproducibility release

Data and code for the paper "When LLM Ensembles Agree They Are Wrong: Convergent Failures in Clinical Question Answering" (anonymous; under double-blind review, ICDM 2026, Applied Data Science track).

The release holds the per-model predictions, the analysis outputs they feed into, the bias labels, the source data behind the figures, and the notebooks we ran to produce all of it.

## Quick start

```bash
pip install -r requirements.txt
python verify_reproducibility.py
```

`verify_reproducibility.py` recomputes the main numbers in the paper directly from the JSON files here and prints PASS or FAIL for each one. It should report `17/17 checks passed` and finishes in well under a minute. No GPU or API key is needed.

## Layout

```
.
├── README.md
├── LICENSE.txt                  MIT for code, CC BY 4.0 for data
├── requirements.txt
├── verify_reproducibility.py    the 17-check script above
├── bundle_release.py            helper to zip the release
├── code/                        16 notebooks: inference, analysis, figures
├── prompts/                     zero-shot and bias-labeling templates
├── predictions/
│   ├── medqa/                   8 per-model files, MedQA-USMLE (1,273 questions)
│   └── medmcqa/                 5 per-model files, MedMCQA (2,816 questions)
├── analysis/                    24 files: audits, figure data, detector and stats output
├── bias_labels/                 7 files: LLM-judge bias labels, programmatic PC scores
└── figures/                     paper figures (PDF and PNG)
```

The notebooks in `code/` are the ones we actually ran, on Colab, with predictions cached to Google Drive, so the paths inside them point there. They are included so the provenance of each file is clear. You do not need them to check the paper's numbers; the JSON files plus `verify_reproducibility.py` are self-contained and run locally.

## Models

| Model | Org | Role |
|---|---|---|
| Flan-T5-base (250M) | Google | near-random baseline |
| Llama-3-8B-Instruct-Lite | Meta | mid-tier |
| Qwen-2.5-7B-Instruct-Turbo | Alibaba | mid-tier |
| Gemma-3n-E4B-it | Google | mid-tier |
| DeepSeek-V3 | DeepSeek | frontier (MedQA only) |
| GPT-4o | OpenAI | frontier (both datasets) |

Llama-3.3-70B and GPT-4o-mini appear only as judges in the bias labeling, not as ensemble members. DeepSeek-V3 was not available on MedMCQA when we ran the experiments, so every MedMCQA result uses the four strong models (Flan, Llama, Qwen, Gemma) with GPT-4o as the frontier. Inference ran on Together.ai (open weights) and the OpenAI API (GPT-4o, GPT-4o-mini) during April and May 2026, zero-shot, greedy decoding at temperature 0 except where a result says otherwise. Permutation tests use seed 42.

## What the verification script covers

The 17 checks are:

- the seven per-model accuracies on MedQA and MedMCQA
- the 5-LLM MedQA unanimous-wrong rate, 14 of 57 (24.6%)
- the MedQA HIGH_RISK detector, n = 102 at 96.1% precision
- the 5-wrong-baseline Fisher exact p-values (0.14 on MedQA, 0.22 on MedMCQA)
- the programmatic premature-closure proxy (Welch t = -2.18, p = 0.030)
- cross-classifier Cohen's kappa on the bias labels (0.126 MedQA, 0.215 MedMCQA)
- the frontier-only baseline: GPT-4o alone reaches 88.5%, and detector routing beats the mid-tier ensemble by 7.7 points
- the MedMCQA strong-to-strong mean Yule Q, 0.633

## Other numbers from the paper

Everything else maps to one file you can open directly.

| Claim | File |
|---|---|
| 5-LLM MedQA unanimous wrong: 14/57 = 24.6% [15.2, 37.1] | `analysis/paper_numbers.json`, `analysis/exp1_holdout_cis.json` |
| 4-LLM MedQA, DeepSeek frontier: 48/143 = 33.6% [26.3, 41.6] | `analysis/audit_empirical_chance.json` |
| 4-LLM MedMCQA, GPT-4o frontier: 71/301 = 23.6% [19.1, 28.7] | `analysis/audit_empirical_chance.json` |
| 3-LLM MedQA mid-tier: 117/279 = 41.9% | `analysis/paper_numbers.json` |
| Detector MedQA HIGH_RISK: n=102 at 96.1% | `analysis/detector_improved_validation.json` |
| Detector MedMCQA looser rule: n=188 at 83.0% | `analysis/detector_medmcqa_validation.json` |
| Detector MedMCQA strict rule (V-F): n=134 at 82.8% | `analysis/exp1_holdout_cis.json` |
| Shuffle: text-lock 60.4%, letter-lock 10.4% (n=48) | `analysis/audit_shuffle_summary.json` |
| k-intersection z-scores (Figure 2) | `analysis/figure2_kintersection_data.json` |
| Vote-share 0.75 gap (Section V-F) | `analysis/exp9_confidence_proxy.json` |
| Pairwise lifts (Table II) | `analysis/audit_pairwise_lift_permutation.json` |
| Yule Q by pair (Section VI-F) | `analysis/exp3_diversity_measures.json` |
| Test-retest Jaccard 0.745 / 0.864 | `analysis/audit_test_retest.json` |
| Bias four-corner table (Table V) | `analysis/triangulation_summary.json`, `bias_labels/` |
| 5-wrong baseline Fisher exact (Section VI-D) | `analysis/audit_5wrong_baseline.json` |
| Programmatic PC (Section VI-G) | `analysis/audit_programmatic_pc.json`, `bias_labels/programmatic_pc_scores.json` |
| Entropy baseline (Section V-F) | `analysis/exp2_entropy_baseline.json`, `analysis/entropy_baseline_full_sweep.json` |
| Frontier substitution (Section VI-C) | `analysis/audit_frontier_substitution.json` |
| Non-dental MedMCQA (Section VI-B) | `analysis/audit_non_dental.json` |
| Frontier-only baseline (Section V-E) | `analysis/frontier_only_baseline.json` |

## The detector

It has no learned parameters and no tuned threshold. For each question, take the three mid-tier models (Llama-3-8B, Qwen-2.5-7B, Gemma-3n) and one frontier model (GPT-4o, or DeepSeek-V3 on MedQA):

- HIGH_RISK: the three mid-tier models give the same answer and the frontier model disagrees.
- CONFIDENT: all four agree.
- DISAGREEMENT: the mid-tier models are not unanimous.

Applying it to new predictions is just checking those three conditions per question.

## Prediction format

Each prediction file is a JSON list. A record looks like:

```json
{"idx": 0, "gold": "B", "pred": "A", "correct": 0}
```

Flan-T5-base files also carry `question` (the stem) and `raw_output` (the raw generation). MedMCQA files carry a UUID `id`.

## Notes on the data

- `analysis/exp9_confidence_proxy.json` was regenerated in June 2026 so it holds the full partition data; an earlier copy was a small metadata stub.
- DeepSeek-V3 returned all-null on MedMCQA at experiment time, so those predictions are not included.
- MedMCQA has two detector variants, a looser rule (n=188, Table IV) and a stricter one (n=134, Section V-F). Both cohorts are released; the V-F footnote explains the difference.
- Cohen's kappa on the per-question bias labels runs from 0.13 to 0.22 across the four dataset-by-classifier corners. The aggregate distributions are stable, but the per-instance labels are not reliable and should not be read as ground truth.

## License

Code (`*.py`) is MIT. Data (predictions, analysis output, bias labels) is CC BY 4.0; see `LICENSE.txt`. MedQA-USMLE and MedMCQA stay under their original licenses, and model outputs are subject to each provider's terms.

## Citation

```bibtex
@inproceedings{anon2026ensembles,
  title  = {When LLM Ensembles Agree They Are Wrong: Convergent Failures in Clinical Question Answering},
  author = {Anonymous},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year   = {2026},
  note   = {Applied Data Science track, under double-blind review}
}
```

## Contact

During review, please reach the authors through the anonymous submission system.
