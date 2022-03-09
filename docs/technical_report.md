# FormulaForge: Execution-Based Grading for Natural-Language-to-Excel-Formula Generation

*Technical report. All numbers are from the checked-in code and the default config (`configs/scratch.yaml`), single training run, seed 0.*

## Abstract
We train a small T5 encoder-decoder from scratch to translate plain-English spreadsheet requests (plus a table schema) into Excel formulas, using a self-generated synthetic dataset of 41.5k examples. Predictions are graded by **executing** them with the `formulas` evaluator on freshly generated held-out tables and comparing results with the gold formula's results. On 1,000 held-out test examples the model reaches **77.8% execution accuracy versus 69.5% exact match**: string matching under-reports accuracy by 8.3 points because many correct formulas are written differently from gold (e.g. `SUMIF` vs `SUMIFS`, equivalent range orderings).

## 1. Dataset card
- **Construction.** 28 templates over 8 families (SUM, AVERAGE, MINMAX, COUNTIF, IF, SUMIFS, VLOOKUP, INDEX_MATCH). Each template has 2-6 NL skeletons; a rule-based paraphraser adds synonym swaps, polite prefixes/suffixes and article dropping. Tables are random schemas of 1-5 columns (text categories, numbers, unique-key columns) with 5-14 rows. Records store a `TableSpec` (schema, size, "value must/must not appear" constraints) rather than cell values, so tables can be regenerated with any seed.
- **Size and splits.** 40,000 train / 500 val / 1,000 test. 23.5k distinct train formulas and 36k distinct train request strings. Train tiers: 9.9k / 14.7k / 15.5k (tier 1/2/3). Tier 1 = single function on one range; tier 2 = one condition, lookup, or partial range; tier 3 = several conditions, nesting, or `IFERROR`.
- **Dedup and leakage control.** Examples are deduplicated on the exact encoder input; train/val/test are disjoint on that input. They are *not* disjoint on template, so the benchmark measures generalisation to new schemas, values and phrasings of known formula shapes.
- **Validity checks.** Every gold formula must evaluate without error on 5 held-out tables; `IFERROR` templates force the key to be absent so the fallback branch runs.
- **Known biases (important).** (a) Key columns (employee/customer/item names) occur *only* in lookup examples (8,044 train rows, all VLOOKUP/INDEX_MATCH), so the model learned "key column present => lookup". On tables with a key column it often emits a lookup for aggregate requests (observed in the demo on the `employees` table). (b) Language is template-derived; real user phrasing is far more varied. (c) Single sheet, exact-match lookups only.

## 2. Model and training
- **Architecture.** `T5ForConditionalGeneration`, random init: 3 encoder + 3 decoder layers, d_model 128, d_ff 384, 4 heads, dropout 0.1; 1.23M parameters.
- **Input/output.** Encoder: `"<request> schema: A Region text; B Revenue number; ...; rows 2 to 11"`. Decoder: the formula without `=`. A regex word-level vocabulary (355 tokens; digits are single tokens; spaces inside quoted strings are protected) replaces T5's sentencepiece vocabulary, which lacks `<`.
- **Optimisation.** AdamW, lr 1.5e-3, weight decay 0.01, 5% warmup then linear decay, batch 64, 8 epochs (5,000 steps), gradient clipping 1.0, seed 0, CPU only (about 45 minutes on 2 cores). Validation loss fell from 0.228 (epoch 1) to 0.029 (epoch 8). Greedy execution accuracy on the first 200 validation examples was 51.5% at epoch 4 and 78.5% at epoch 8.
- **Decoding.** Beam search (4 beams) by default.

## 3. Evaluation protocol
For each example the gold and predicted formulas are compiled once and evaluated on *n* tables regenerated from the example's spec with new seeds. A prediction is execution-correct only if its result equals gold's on **every** table (numbers within 1e-6, strings case-insensitively, as in Excel). Parse errors, unknown functions and Excel error values never count as correct, even if gold and prediction both fail.

## 4. Results (test, n = 1000, 4 beams, 3 tables)
| Tier | N | Exact match | Execution accuracy |
|---|---|---|---|
| 1 | 264 | 88.6% | 89.8% |
| 2 | 382 | 66.5% | 73.6% |
| 3 | 354 | 58.5% | 73.4% |
| **All** | 1000 | **69.5%** | **77.8%** |

By family (execution accuracy): COUNTIF 92.7, IF 87.1, MINMAX 84.2, SUM 82.2, AVERAGE 80.6, VLOOKUP 73.1, INDEX_MATCH 67.0, SUMIFS 57.9. The gap between metrics is largest for SUMIFS (+17.5 pts), VLOOKUP (+14.0), INDEX_MATCH (+11.7) and SUM (+13.2). Tiers are template-structure labels and not strictly ordered by difficulty (tier-3 IFERROR lookups are easier than tier-2 plain lookups). Failure modes among the 222 errors: wrong reference 85.1% (valid formula, wrong column/range), wrong operator or number 12.2%, wrong function 1.8%, invalid formula 0.9%. Full tables: `docs/sample_report_test.md`.

## 5. Ablations
Only evaluation-time ablations were run (same trained model; no retraining). Training-side ablations (model size, data size, paraphrasing on/off, from-scratch vs. t5-small) are **not** reported because they were not run.

| Setting | Exact match | Execution accuracy |
|---|---|---|
| Greedy, 1 held-out table | 70.4% | 80.3% |
| Greedy, 3 tables | 70.4% | 78.4% |
| Beam 4, 1 table | 69.5% | 79.7% |
| Beam 4, 3 tables (default) | 69.5% | 77.8% |
| Beam 4, 5 tables | 69.5% | 77.0% |

Findings: (i) grading on a single table overstates accuracy by 2-3 points, because a wrong formula can coincide with gold on one random table; more tables tighten the estimate (still decreasing at 5). (ii) Beam search did not help; greedy was 0.6 points better in execution accuracy at 3 tables (within noise for n = 1000). The default config keeps 4 beams for historical reasons; pass `--beams 1` to use greedy.

## 6. Limitations
Single seed and no confidence intervals (with n = 1000 the standard error is about 1.3 points); a tiny model trained for a short schedule; synthetic, template-derived language; the key-column shortcut described in Section 1; the `formulas` library's Excel-compatibility quirks carry over into grading; the fixed-vocabulary tokenizer maps unseen words to `<unk>`.
