# Earned Pass

A receiving screen for ground cinnamon, built for the John Deere hackathon crop track from the
"Implementation plan: Earned Pass" section of the Team Hub doc (Agent A tab).

It reads a colorimeter value and a photo and returns **PASS**, **LAB** or **HOLD**. LAB and HOLD both end in a
lab check, so PASS is the only lane that removes one, and the only lane that has to be earned: it is issued
only when two claims are certified on held-out evidence, counted in independent units.

Scope: lab-prepared blends, three fillers (peanut shell, walnut shell, wood dust), 0 to 24% w/w. Not validated
on phone images. Allergen presence is not measured. No detection limit, speed or price is claimed.

## Run it

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m earnedpass.run_all      # about 2 minutes the first time, 30 seconds after
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest tests
```

The dataset is read in place from `../Multimodal Dataset for Cinnamon Powder Adulteratio/Dataset`.
Set `EARNEDPASS_DATA` to point somewhere else.

## The locked cells

Levels 3, 10 and 18 of every filler (9 cells, 450 samples) are locked. `run_all` never trains on them or
scores them, and until they are scored the Inspect page draws from the 30 open cells, each judged by a model
that never saw its cell.

Sunday 8 AM, once:

```bash
.venv/bin/python -m earnedpass.run_all --score-locked
```

This scores the locked cells with the model trained on the 30 open cells, adds them to the registry and
switches Inspect over to them. A second run is refused unless you pass `--force`.

## What is where

| File | Job |
|---|---|
| `earnedpass/load.py` | One row per sample ID. Labels come from the ID (`^C([PWD])(\d+)S(\d+)$`), never the category column. Contract checks. |
| `earnedpass/groups.py` | Twin photos (`data/image_twins.csv`), repeated colour readings, photo sessions. |
| `earnedpass/features/` | Colour (L\*, a\*, b\*, chroma, hue, ΔE to reference), FTIR (−log10, Savitzky-Golay derivative), image (thumbnail, dish crop, colour statistics, hue histogram, Laplacian variance). |
| `earnedpass/models.py` | Feature blocks and models, fitted inside each fold. Seven sensor sets are block masks. |
| `earnedpass/eval/splits.py` | The four split rungs. Asserts no sample ID (or twin, cell, folder) sits on both sides. |
| `earnedpass/eval/ladder.py` | Stress ladder: every sensor set and task on every rung. |
| `earnedpass/eval/conformal.py`, `bounds.py` | Conformal band sets; Clopper-Pearson bounds and the lots-needed calculator. |
| `earnedpass/decide.py` | Lanes, the two PASS claims, the licence, the B1 forced-call baseline, the blind-spot map, reason sentences. |
| `earnedpass/audit.py`, `controls.py` | Replicate Radar; session probe, zero-filler probe, offset test, band check. |
| `earnedpass/run_all.py` | Runs everything and writes `artifacts/results.json`, the numbers registry. |
| `app.py` | Streamlit app: Inspect, Evidence, Log. |

Every number for a slide should be read from `artifacts/results.json` (`rows` has task, sensor set, rung, mode,
metric, value, sample count and cell count).

## Choices made where the plan left room

- **Locked cells are out of every Saturday number.** The ladder, calibration, licence and controls use the 30
  open cells only, so rung 3 has 30 folds and the catch claim counts 18 cells (24 once the locked cells are scored).
- **Colour block has six features**, not five: ΔE to the reference is included. The reference is the mean of the
  0% samples from the same folder that the fold's model may see; if there are none it is the mean of the other folders.
- **PASS claims are checked on PASS candidates** (band set equal to {zero}) before the licence is applied, since
  the licence depends on the claims.
- **A wrong HOLD is a HOLD on a 0% sample.** A HOLD on a low-level blend is not counted as wrong.
- **B1** forces a call from the top band: 0% band means PASS, anything else HOLD.
- **Strict counting is measured, not offered as a mode.** The Radar reports 22 distinct mean FTIR spectra and
  7 photo days, but the mode switch has two settings.
- **The threshold fallback** is computed and stored in `results.json` (`decision.<set>.threshold_fallback`) but
  not shown in the app.
- Left out, as the plan's stretch items: pretrained CNN features, the turmeric audit, simulated lots.

## Still assumed

FTIR values are transmittance; what the 0% folders hold; the cinnamon species.
