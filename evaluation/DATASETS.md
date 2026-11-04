# Evaluation datasets

The frozen question sets are derived from DrugCentral content and are **not included** in this repository.
Generate them with the builders below, or obtain the original files separately and check them against these
SHA-256 values.

| Set (path) | SHA-256 of `questions.json` | Builder |
|---|---|---|
| `evaluation/fresh-240-20260928/` (used for the reported verified benchmark) | `24620c6478bfdab2857b779bc9b7cd964ce6b8a5a56264d483b91acea428b8ca` | `python -m evaluation.build_fresh_questions --output evaluation/fresh-240-20260928/questions.json` (seed 20260928) |
| `evaluation/complex-110/` (`manifest.json` is tracked) | `92f72549638b4bb396687ea504e3698cacfd155b643187a54613753bf60d459c` | `python -m evaluation.build_complex_questions --output evaluation/complex-110` (seed 20261004, needs `data/labels.sqlite`) |
| `evaluation/full-schema-300-drugcen-2027-20261003/` | `def09d496992bcd373a6533a3e1a8aa370ba4c9ef1934449f8b65db2095cf188` | `python -m evaluation.build_full_table_questions --source SOURCE.sql.gz --output …` (seed 20261003) |
| `evaluation/user-style-new-tables/` | `1937195093fe4ac2d973d4910e2d0902495731da46d57aef2c6761d9eb54a32c` | `python -m evaluation.build_user_style_questions --output evaluation/user-style-new-tables` (seed 20261008) |
| `evaluation/fresh-240-20260928/baseline-v4-answers.json` (baseline answers for `compare_runs`) | `0074ad00b1b8c05522af3ca9a589b06292f5cf781fb8d510cc7b0dcab8ab9163` | Saved answers from an earlier run; not regenerable |

Reproducibility limits:

* `build_fresh_questions` excludes questions listed in `benchmarks/web-judge-240/human_review.json` (its default
  `--exclude`), which is not part of this repository. Without that file the regenerated fresh-240 set will differ,
  so an exact reproduction of the reported benchmark needs the original `questions.json` (hash above).
* Builders read the configured source database or a DrugCentral dump; results depend on the DrugCentral release.

Reported verified benchmark (fresh-240, 2026-10-20, full-coverage collections `…_20261010`): run with
`python -m evaluation.verified_benchmark --output benchmarks/<run>` and judged with
`python -m evaluation.judge_verified` (Amazon Nova Lite, `us.amazon.nova-lite-v1:0`): retrieval success 100.0%,
faithfulness 100.0%, factual correctness 100.0% on 240 questions. A separate external review of the same answers
with a GPT model (outside this repository) scored factual correctness 97.5% (6 of 240 incorrect). Result files
are not included.
