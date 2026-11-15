# Earned Pass

A receiving screen for ground cinnamon, built for the CDA Precision & Digital Agriculture Hackathon (John Deere,
Track 4, Challenge 2). It supplements lab testing; it does not replace it.

For each sample, the screen compares the colorimeter reading and the photo with **the supplier's own reference
sample**, estimates the chance that each filler is present, and gives one verdict:

| Verdict | When (default limits, adjustable in the sidebar) |
|---|---|
| ✕ **Reject** | chance of wood dust above 10%: discard the lot |
| ⚠ **Lab (wood risk)** | chance of wood dust above 1%: confirm at the lab before any use |
| 🏷 **Label** | chance of peanut shell or walnut shell above 10%: label the lot. Both can be labelled; the screen does not choose between them |
| ⧖ **Lab (verify pure)** | nothing above its limit: the lab verifies the lot is pure |

Wood decides the verdict when it fires; nut labels are still reported alongside. The rules come from the team's
Mentor Notes (Sat Nov 14) and live in `earnedpass/verdict.py`.

**Chance present** = chance of any filler (band model: 1 − P(0% band)) × chance it is that filler, given there is
one (filler model). The multiplication matters: the filler model always splits 100% between the three fillers, so
on its own it could never send a lot to "verify pure".

Scope: lab-prepared blends, three fillers (peanut shell, walnut shell, wood dust), 0 to 24% w/w. Not validated on
phone images. Filler is measured, allergen content is not. No detection limit, speed or price is claimed.

## Run it

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m earnedpass.run_all      # about 2 minutes the first time, 30 seconds after
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest tests
```

The dataset is read in place from `../Multimodal Dataset for Cinnamon Powder Adulteratio/Dataset`.
Set `EARNEDPASS_DATA` to point somewhere else.

## The app

| Page | What it shows |
|---|---|
| Inspect | One sample: verdict card, chance of each filler against its limit, amount estimate, ground truth with a right/wrong check, colour readings. Record your own decision. |
| Compare | Two samples side by side. |
| Scoreboard | Fair-test results: wood let through, filler missed, nut labels, pure sent to verify, the full verdict table, easy-vs-fair, and the locked-cell final test. |
| Evidence | Why the supplier reference is used, blind-spot map, Replicate Radar, stress ladder, sanity checks. |
| Log | Every recorded decision with the limits in force. |

## How results are tested

- **Fair test:** every sample is judged by models trained without its whole material (one filler at one level,
  50 samples), so the model never sees near-copies of what it judges.
- **Supplier reference:** half of each folder's pure batch (24 samples, split so no identical photo is on both
  sides) is that supplier's reference and is never screened. The other 26 are screened.
- **Locked cells:** levels 3, 10 and 18 of every filler (9 materials, 450 samples) are never used for training,
  tuning or any number until the one-time final run:

```bash
.venv/bin/python -m earnedpass.run_all --score-locked
```

A second run is refused unless you pass `--force`.

## Known limits

- Each filler was blended into its own cinnamon batch and photographed on its own days, so telling the fillers
  apart may partly be recognising the batch. There is no blend of a filler into another batch to test this.
- The pure samples screened share a batch with their supplier's reference, so pure-cinnamon results are optimistic.
- A new supplier needs a reference sample. Compared with pure cinnamon in general, unfamiliar pure cinnamon is
  mistaken for wood (Evidence → Why supplier reference).
- A filler the models have never seen (cassia, lead chromate, other dust) would still be named as one of the three.

## What is where

| File | Job |
|---|---|
| `earnedpass/load.py` | One row per sample ID. Labels come from the ID (`^C([PWD])(\d+)S(\d+)$`), never the category column. Contract checks. |
| `earnedpass/groups.py` | Twin photos (`data/image_twins.csv`), repeated colour readings, photo sessions. |
| `earnedpass/features/` | Colour (L\*, a\*, b\*, chroma, hue, ΔE to reference), FTIR (−log10, Savitzky-Golay derivative), image (thumbnail, dish crop, colour statistics, hue histogram, Laplacian variance). |
| `earnedpass/reference.py` | Supplier-reference mode: the reference split and features as differences from the reference. |
| `earnedpass/models.py` | Feature blocks and the three models (band, filler, amount), fitted inside each fold. Seven sensor sets are block masks. |
| `earnedpass/verdict.py` | The team's verdict rules, chance-present calculation, right/wrong judging and summary numbers. |
| `earnedpass/eval/splits.py`, `ladder.py` | Split rungs and the stress ladder (easy vs fair). |
| `earnedpass/audit.py`, `controls.py` | Replicate Radar; session probe, zero-filler probe, offset test, band check. |
| `earnedpass/run_all.py` | Runs everything and writes `artifacts/results.json`, the numbers registry. |
| `app.py` | The Streamlit app. |

`earnedpass/decide.py`, `eval/conformal.py`, `eval/bounds.py` and `scoreboard.py` belong to the earlier PASS/LAB/HOLD
design. The app no longer uses them; `run_all.py` still writes their results into `results.json`.

## Still assumed

FTIR values are transmittance; what the 0% folders hold; the cinnamon species.
