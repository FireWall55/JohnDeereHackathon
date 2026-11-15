# Earned Pass

A receiving screen for ground cinnamon, built for the CDA Precision & Digital Agriculture Hackathon (John Deere,
Track 4, Challenge 2). It supplements lab testing; it does not replace it.

For each sample, the screen compares the colorimeter reading and the photo with **the supplier's own reference
sample**, estimates the chance that each filler is present, and gives one verdict:

| Verdict | When (default limits, adjustable in the sidebar) |
|---|---|
| ✕ **Reject** | chance of wood dust above 2%: discard the lot |
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

## Screen your own samples (no dataset needed)

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py
```

Open **Screen a sample**. The saved model in `models/screen_model.pkl` comes with the repo, so you only need:

1. **A reference for each supplier** (Suppliers tab): several photos and several colorimeter readings (L\*, a\*, b\*)
   of that supplier's pure cinnamon, confirmed pure by a lab. The screen averages them, so more of each gives a
   steadier reference. Three example suppliers from the dataset are built in. Your suppliers are saved locally in
   `user_data/suppliers.json` (not committed).
2. **Each sample:** one photo and one colorimeter reading, screened on its own or as a batch (a CSV with
   `sample, supplier, L, a, b, photo` plus the photos; a template is in the app).

`examples/` has photos you can try: `example_batch.csv` with its eight photos for the batch tab, and
`reference_photos/` (three photos plus `readings.csv`) for registering a supplier. They come from the materials the
model was trained on, so they show how to use the screen, not how accurate it is.

**Photos must look like the dataset's:** a top-down photo of the powder in a round dish on a light background, from
the original camera file. The screen measures inside 70% of the dish and shows the measured area. A sample further
from the training data than 99.9% of training samples is marked **outside the training range** and sent to the lab;
recompressed or edited photos, a different camera or lighting, a differently set-up colorimeter, or an unknown filler
can all trigger this.

## Rebuild everything from the dataset

```bash
.venv/bin/python -m earnedpass.run_all      # fair-test results, about 2 minutes the first time
.venv/bin/python -m earnedpass.deploy       # the screening model in models/screen_model.pkl
.venv/bin/python -m pytest tests
```

The dataset is read in place from `../Multimodal Dataset for Cinnamon Powder Adulteratio/Dataset`.
Set `EARNEDPASS_DATA` to point somewhere else. The screening model is trained on every open material in
supplier-reference mode with colorimeter + photo; the locked materials are never used.

## The app

What an inspector sees:

| Page | What it shows |
|---|---|
| Screen a sample | Upload a photo and a colour reading (or a batch), pick the supplier, get the verdict. Register suppliers' references. |
| Demo samples | Samples from the hackathon dataset, one at a time (Previous / Next), each judged by models that never saw that material. Only when the dataset is present. |
| How it works | The workflow, why every sample is compared with its supplier's reference, and how to bring on a new supplier or a new kind of cinnamon. |
| Log | Every recorded decision with the limits in force. |

**Developer view** (switch at the bottom of the sidebar, off by default) adds what only the team and the judges
should see: ground truth and a "Was the screen right?" card on Inspect, demo controls, notes on how far to trust the
result, the held-out-supplier results on How it works, the true material in the Log, and three extra pages:

| Page | What it shows |
|---|---|
| Compare | Two samples side by side, chosen by their true material. |
| Scoreboard | Fair-test results: wood let through, filler missed, nut labels, pure sent to verify, the full verdict table, easy-vs-fair, and the locked-cell final test. |
| Evidence | Why the supplier reference is used, blind-spot map, Replicate Radar, stress ladder, sanity checks. |

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
| `earnedpass/deploy.py`, `models/screen_model.pkl` | The saved screening model for new samples: photo processing, supplier references, the outside-training-range check. |
| `examples/` | Photos and tables for trying Screen a sample without the dataset. |
| `earnedpass/eval/splits.py`, `ladder.py` | Split rungs and the stress ladder (easy vs fair). |
| `earnedpass/audit.py`, `controls.py` | Replicate Radar; session probe, zero-filler probe, offset test, band check. |
| `earnedpass/run_all.py` | Runs everything and writes `artifacts/results.json`, the numbers registry. Its `verdicts` section holds the default-limit verdict numbers for every instrument set, in supplier-reference and general mode. |
| `app.py`, `.streamlit/config.toml` | The Streamlit app and its light and dark themes. |

## Still assumed

FTIR values are transmittance; what the 0% folders hold; the cinnamon species.
