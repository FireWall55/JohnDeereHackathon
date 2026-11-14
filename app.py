"""Earned Pass: a receiving screen for ground cinnamon.

    streamlit run app.py

Reads artifacts written by `python -m earnedpass.run_all`.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from earnedpass import config as C
from earnedpass import data as data_mod
from earnedpass import decide, models
from earnedpass.eval.bounds import cp_lower, units_needed
from earnedpass.eval.conformal import min_alpha
from earnedpass.eval.ladder import HEADLINE
from earnedpass.features import color
from earnedpass.run_all import PROBS, RESULTS

LOG = C.ARTIFACTS / "decision_log.csv"
LOG_COLUMNS = ["time", "sample_id", "model_version", "mode", "must_catch", "alpha", "pass_bar", "sensors",
               "recommended", "decision", "override", "sent_to_lab", "user"]

# Status colours always travel with an icon and a word, never alone.
LANE = {
    "PASS": {"icon": "✓", "colour": "#0ca30c", "text": "no lab check for this sample"},
    "LAB": {"icon": "⧖", "colour": "#fab219", "text": "routine lab confirmation"},
    "HOLD": {"icon": "⚠", "colour": "#d03b3b", "text": "strong suspicion: the lot waits and its lab check goes first"},
}
FILLER_COLOUR = {"P": "#2a78d6", "W": "#eb6834", "D": "#1baf7a"}
SEQ = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
SET_LABEL = {"colour": "Colorimeter", "ftir": "FTIR", "image": "Photo", "colour+ftir": "Colorimeter + FTIR",
             "colour+image": "Colorimeter + photo", "ftir+image": "FTIR + photo", "all": "All three"}
UNIT_WORD = {"as_labelled": "samples", "replicate_aware": "independent cells"}

st.set_page_config(page_title="Earned Pass", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading artifacts...")
def load(_stamp):
    d = data_mod.get()
    results = json.loads(RESULTS.read_text())
    z = np.load(PROBS)
    probs = {ss: decide.Probs(z["p1"][i], z["p3"][i], z["pl"][i]) for i, ss in enumerate(z["sets"])}
    types = {ss: (z["t3"][i], z["tl"][i]) for i, ss in enumerate(z["sets"])} if "t3" in z else None
    levels = {ss: (z["e3"][i], z["el"][i]) for i, ss in enumerate(z["sets"])} if "e3" in z else None
    return d, results, probs, models.Spectra(d.X_ftir), types, levels


@st.cache_resource(show_spinner=False)
def evaluate(_stamp, mode, alpha, must_catch, bar, sensor_set):
    d, _, probs, _, _, _ = load(_stamp)
    return decide.evaluate(d.samples, probs[sensor_set], decide.Settings(mode, alpha, must_catch, bar, sensor_set))


if not (RESULTS.exists() and PROBS.exists()):
    st.error("No results yet. Build them first:")
    st.code("python -m earnedpass.run_all", language="bash")
    st.stop()

STAMP = RESULTS.stat().st_mtime
D, R, PROBS_BY_SET, SPECTRA, TYPES, LEVEL_EST = load(STAMP)
S = D.samples
SCORED = bool(R["meta"]["locked_scored_at"])

MODE_LABEL = {"replicate_aware": "Careful: count each material once", "as_labelled": "Trusting: count every sample"}
BAND_NAME = {"zero": "none (0%)", "low": "low (1–4%)", "mid": "medium (6–12%)", "high": "high (15–24%)"}

with st.sidebar:
    st.header("Screen settings")
    st.caption("A QC manager sets these once, before the screen is used. The starting values are examples, not legal limits.")
    EXPLAIN = st.toggle("Explain each setting", True)

    def note(text):
        if EXPLAIN:
            st.caption(text)

    MODE = st.radio("How to count repeated measurements", list(MODE_LABEL), format_func=MODE_LABEL.get,
                    help="Technical names: replicate-aware and as labelled. A material (one filler at one level) is called a cell.")
    note("Every material in the dataset was measured 50 times. **Careful** treats those 50 as one piece of evidence, because "
         "they are repeats of the same powder. **Trusting** treats them as 50, which makes the screen look better proven "
         "than it is.")

    MUST = st.select_slider("Smallest adulteration that must never slip through", decide.MUST_CATCH_OPTIONS,
                            value=C.MUST_CATCH, format_func=lambda v: f"{v}% filler",
                            help="Technical name: must-catch level, in % by weight.")
    note("A sample gets **HOLD** only when the screen is sure it contains at least this much filler. "
         "Slide left to be stricter: more samples are held.")

    CONF = st.slider("How sure the screen must be", 70, 98, int(round((1 - C.ALPHA) * 100)), 1, format="%d%%",
                     help="Technical name: 1 − α, the coverage of the conformal band set.")
    ALPHA = round(1 - CONF / 100, 2)
    note(f"The screen keeps a shortlist of adulteration ranges it cannot rule out. At {CONF}%, the true range should be on "
         f"that shortlist about {CONF} times in 100. Slide right to be more cautious: longer shortlists, more samples "
         "sent to the lab.")

    BAR = st.slider("Proof needed before PASS is allowed", 50, 99, int(round(C.PASS_BAR * 100)), 1, format="%d%%",
                    help="Technical name: PASS bar. Both PASS claims need a certified lower bound at or above it.") / 100
    note("PASS lets a sample skip the lab, so it has to be earned. The screen may only say PASS once its track record on "
         "materials it has never seen is proven to at least this level. Slide right to make PASS harder to earn.")

    SENSORS = st.selectbox("Instruments used for screening", list(C.SENSOR_SETS), index=list(C.SENSOR_SETS).index(C.SCREEN_SET),
                           format_func=SET_LABEL.get)
    note("A colorimeter and a camera are what an inspector has at the receiving dock. FTIR is a lab instrument.")

    USER = st.text_input("Your name", "inspector-1")
    note("Written into the decision log next to every decision you record.")
    st.divider()
    st.caption("The tool recommends; the inspector decides.")
    st.caption(f"Model {R['meta']['model_version']} · " + ("locked cells scored " + R["meta"]["locked_scored_at"] if SCORED
               else "locked cells (levels 3, 10, 18) not scored yet"))

RES = evaluate(STAMP, MODE, ALPHA, MUST, BAR, SENSORS)
LAB_RES = evaluate(STAMP, MODE, ALPHA, MUST, BAR, C.LAB_SET)


def pct(x, digits=1):
    return f"{x:.{digits}%}"


def chart(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **kw)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def lane_card(lane, body):
    m = LANE[lane]
    st.markdown(
        f"""<div style="border-left:6px solid {m['colour']};padding:.6rem 1rem;border-radius:6px;
        background:color-mix(in srgb,{m['colour']} 9%,transparent)">
        <div style="font-size:1.9rem;font-weight:700;line-height:1.2">{m['icon']} {lane}</div>
        <div style="opacity:.8">{m['text']}</div><div>{body}</div></div>""",
        unsafe_allow_html=True,
    )


def cell_name(cell: str) -> str:
    """'CW6' -> 'Walnut shell, 6%'. Also accepts a full sample ID such as 'CW6S24'."""
    code, level = cell[1], cell[2:].split("S")[0]
    return f"0% ({C.FOLDER_LABEL[code]} folder)" if level == "0" else f"{C.FOLDER_LABEL[code]}, {level}%"


def describe(sample) -> str:
    if sample.level == 0:
        return f"{sample.sample_id}: 0%, the unblended material in the {C.FOLDER_LABEL[sample.code]} folder"
    return f"{sample.sample_id}: {C.FOLDER_LABEL[sample.code]}, {sample.level}% w/w"


def pass_status():
    """Plain-language banner: is PASS available, and if not, why."""
    lic, cl = RES.licence, RES.claims
    unit = lic["unit"]
    if lic["licensed"]:
        st.success(f"✓ **PASS is available.** The screen has proven both of its PASS promises to at least {pct(BAR, 0)}.")
        return
    ref, catch = cl["reference"][unit], cl["catch"][unit]
    what = "samples" if unit == "samples" else "materials"
    st.warning(
        f"🔒 **PASS is switched off at these settings, so no sample will get it.** PASS lets a sample skip the lab, so the "
        f"screen must first prove two promises on materials it has never seen, each to at least {pct(BAR, 0)}:\n\n"
        f"1. *It never passes a sample with {MUST}% filler or more.* It kept this for {catch['k']:,} of {catch['n']:,} "
        f"{what}, which proves **{pct(catch['bound'])}**. {'✓' if catch['bound'] >= BAR else '✗'}\n"
        f"2. *It recognises a batch of pure cinnamon it has not seen before.* It managed this for {ref['k']:,} of "
        f"{ref['n']:,} {what}, which proves **{pct(ref['bound'])}**. {'✓' if ref['bound'] >= BAR else '✗'}\n\n"
        "The dataset has only three batches of pure cinnamon and they look different from each other, so pure cinnamon from "
        "a new batch looks adulterated to the screen. Until that is fixed with more pure-cinnamon evidence, every sample "
        "goes to **LAB** or **HOLD**. See Evidence → PASS licence for the numbers.")


FILLER_SURE = 0.80  # below this the screen says it cannot tell which filler


def filler_guess(cur, pool, row):
    """Which of the three known fillers the screen thinks is in the sample."""
    if TYPES is None:
        st.caption("Filler type is not available yet. Run `python -m earnedpass.run_all` once to add it.")
        return
    t3, tl = TYPES[SENSORS]
    p = (tl if pool == "locked" else t3)[cur]
    codes = list(C.FOLDERS)
    top = int(np.argmax(p))
    blends = D.dev[S.level.to_numpy()[D.dev] > 0]
    hit = float((t3[blends].argmax(1) == S.code.iloc[blends].map(codes.index).to_numpy()).mean())
    lead = "**If there is filler, most likely:**" if row.in_zero else "**Likely filler:**"
    if row.candidate == "PASS":
        st.markdown("**Likely filler:** none suspected")
    elif p[top] < FILLER_SURE:
        st.markdown("**Likely filler:** cannot tell which. " + ", ".join(f"{C.FOLDER_LABEL[c]} {pct(v, 0)}" for c, v in zip(codes, p)))
    else:
        nut = " Possible nut-shell material: confirm at the lab." if codes[top] in ("P", "W") else ""
        st.markdown(f"{lead} {C.FOLDER_LABEL[codes[top]]} (model {pct(p[top], 0)}).{nut}")
    with st.popover("How far to trust the filler guess"):
        st.markdown(
            f"- The screen only knows three fillers: peanut shell, walnut shell and wood dust. It always picks among them, "
            "so a filler it has never seen would be given one of these names.\n"
            f"- On blends it had not seen, these instruments named the right filler for **{pct(hit)}** of samples.\n"
            "- In this dataset each filler was prepared and photographed in its own batch, on its own days. The guess may "
            "partly be recognising the batch, not the filler.\n"
            "- It has no answer for “no filler”. If the screen flags pure cinnamon by mistake, it will still name a filler "
            "(see Evidence → Controls → Zero-filler probe).\n"
            "- It says nothing about allergens. Allergen content is not measured.")


def append_log(row: dict):
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


# ---------------------------------------------------------------- Inspect
BAND_TOP = {"zero": 0, "low": 4, "mid": 12, "high": 24}


def level_estimate(cur, pool):
    """(estimate in %, margin in points) or None. The margin covers 8 in 10 held-out samples."""
    if LEVEL_EST is None:
        return None
    e3, el = LEVEL_EST[SENSORS]
    est = float((el if pool == "locked" else e3)[cur])
    dev = D.dev
    margin = float(np.quantile(np.abs(e3[dev] - S.level.to_numpy()[dev]), 0.8))
    return est, margin


def estimate_text(cur, pool) -> str:
    le = level_estimate(cur, pool)
    if le is None:
        return "not available: run `python -m earnedpass.run_all`"
    est, m = le
    lo, hi = max(0, est - m), min(24, est + m)
    return f"about {est:.0f}%  (likely {lo:.0f}–{hi:.0f}%)"


def amount_text(row) -> str:
    """The filler amount the screen cannot rule out, as one range."""
    kept = [b for b in C.BANDS if row[f"in_{b}"]]
    if not kept:
        return "cannot tell"
    lo, hi = C.BAND_FLOOR[kept[0]], BAND_TOP[kept[-1]]
    if hi == 0:
        return "0% (no filler)"
    return f"{lo}–{hi}% ({' or '.join('none' if b == 'zero' else BAND_NAME[b].split(' ')[0] for b in kept)})"


def truth_checks(cur, pool, row, smp) -> dict:
    """The dataset's own label for this sample, and how the screen did against it."""
    truth = "pure cinnamon (0%)" if smp.level == 0 else f"{C.FOLDER_LABEL[smp.code]}, {smp.level}%"
    if row.lane == "LAB":
        call = "no call (left to the lab)"
    elif row.lane == "PASS":
        call = "✓ right" if smp.level == 0 else "✗ wrong: adulterated sample passed"
    else:
        call = "✗ wrong: pure cinnamon held" if smp.level == 0 else "✓ right: it is adulterated"
    guess = filler_short(cur, pool, row)
    if TYPES is None or guess == "cannot tell which":
        filler = "no guess made"
    elif guess == "none suspected":
        filler = "✓ right" if smp.level == 0 else "✗ wrong: there is filler"
    elif smp.level == 0:
        filler = "there is no filler (the screen had not ruled that out)" if row.in_zero else "✗ wrong: there is no filler"
    else:
        filler = "✓ right" if guess.startswith(C.FOLDER_LABEL[smp.code]) else "✗ wrong"
    out = {"Ground truth": truth, "Recommendation was": call, "Filler guess was": filler}
    le = level_estimate(cur, pool)
    if le is not None:
        est, m = le
        off = abs(est - smp.level)
        out["Amount estimate was"] = (f"{'✓' if off <= m else '✗'} off by {off:.0f} points"
                                      + ("" if off <= m else " (outside its likely range)"))
    return out


def page_inspect():
    st.title("Sample screening")
    pool = "locked" if SCORED else "dev"
    df = RES.df[RES.df.pool == pool].set_index("row")
    pick = st.session_state.get("demo_pick", "Any material")

    def draw():
        choices = df.index[df.cell == pick] if pick in set(df.cell) else df.index
        st.session_state.cur = int(np.random.default_rng().choice(choices))
        st.session_state.revealed = None

    if st.session_state.get("cur") not in df.index:
        draw()
    a, b = st.columns([1, 1.5], gap="large", vertical_alignment="center")
    a.button("Next sample", on_click=draw, type="primary")
    show_truth = b.toggle("Show ground truth", True, help="The dataset's own label for the sample, and whether the screen "
                          "got it right. Turn off to screen blind; it then appears after you record a decision.")

    cur = st.session_state.cur
    row, smp = df.loc[cur], S.iloc[cur]
    revealed = st.session_state.get("revealed") == cur
    lab = LAB_RES.df.set_index("row").loc[cur]

    # ---- first glance: the result, the filler, the amount
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.image(str(C.THUMBS / f"{smp.sample_id}.jpg"), caption=f"Sample #{cur:04d}")
    with right:
        lane_card(row.lane, "")
        rows = {"Likely filler": filler_short(cur, pool, row), "Estimated filler amount": estimate_text(cur, pool)}
        if show_truth or revealed:
            rows.update(truth_checks(cur, pool, row, smp))
        else:
            rows["Ground truth"] = "hidden until you record a decision"
        rows.update({"L* (lightness)": f"{smp.L:.1f}", "a* (green to red)": f"{smp.a:.1f}", "b* (blue to yellow)": f"{smp.b:.1f}",
                     "ΔE to its reference": f"{reference_delta(cur, pool):.1f}"})
        st.dataframe(pd.DataFrame({"This sample": rows}), width="stretch")
        if not RES.licence["licensed"]:
            st.caption("🔒 PASS is switched off at these settings. The reason is under “Details and reasoning”.")

    # ---- the inspector's decision
    a, b = st.columns([1, 1.5], gap="large", vertical_alignment="bottom")
    decision = a.radio("Your decision", list(LANE), index=list(LANE).index(row.lane), horizontal=True, key=f"decision{cur}",
                       format_func=lambda k: f"{LANE[k]['icon']} {k}", disabled=revealed)
    label = "Record and release without a lab check" if decision == "PASS" else "Record and send to lab"
    if b.button(label, disabled=revealed):
        append_log(dict(time=dt.datetime.now().isoformat(timespec="seconds"), sample_id=smp.sample_id,
                        model_version=R["meta"]["model_version"], mode=MODE, must_catch=MUST, alpha=ALPHA, pass_bar=BAR,
                        sensors=SENSORS, recommended=row.lane, decision=decision, override=decision != row.lane,
                        sent_to_lab=decision != "PASS", user=USER))
        if decision != "PASS":
            st.session_state.setdefault("queue", []).append(
                dict(sample=f"#{cur:04d}", lane=decision, p_must_catch=float(row.p_must_catch)))
        st.session_state.revealed = cur
        st.rerun()
    if decision != row.lane and not revealed:
        st.warning(f"This overrides the recommendation ({row.lane}). The override is recorded under your name.")

    if revealed:
        last = pd.read_csv(LOG).query("sample_id == @smp.sample_id").iloc[-1]
        if last.decision == "PASS" and smp.level > 0:
            st.error("An adulterated sample was released without a lab check. This is the error the PASS rules exist to prevent.")
        elif last.decision == "HOLD" and smp.level == 0:
            st.warning("Pure cinnamon was held: a delay and a lab fee, and the lab check clears it.")
        elif last.decision == "PASS":
            st.success("Pure cinnamon was released without a lab check.")
        else:
            st.success("Sent to the lab, which settles this sample.")

    # ---- everything else, out of the way
    with st.expander("Details and reasoning"):
        st.markdown(f"**Why {row.lane}:** {decide.reason(row, RES)}")
        fig = go.Figure(go.Bar(
            y=[BAND_NAME[b] + ("  ● not ruled out" if row[f"in_{b}"] else "") for b in C.BANDS][::-1],
            x=[row[f"p_{b}"] for b in C.BANDS][::-1], orientation="h", marker_color="#2a78d6", width=0.5,
            text=[pct(row[f"p_{b}"], 0) for b in C.BANDS][::-1], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        chart(fig, 190, title="How likely each filler level is, according to the model",
              xaxis=dict(range=[0, 1.12], tickformat=".0%", showgrid=False))
        st.caption("The model is often very sure and still wrong on materials it has not seen, so the screen keeps every "
                   "level it cannot rule out, not just the most likely one.")
        st.markdown(f"**Filler levels the screen cannot rule out:** {amount_text(row)}. This range, not the estimate, "
                    "decides PASS, LAB or HOLD." + (f" It {'contains' if row.covered else 'does not contain'} the true amount."
                                                    if show_truth or revealed else ""))
        le = level_estimate(cur, pool)
        if le is not None:
            st.caption(f"The estimated amount comes from a separate model that predicts a single number. On materials it "
                       f"had not seen, its estimate was within ±{le[1]:.0f} points of the truth for 8 samples in 10. It can "
                       "disagree with the range above; when it does, trust the range.")
        filler_guess(cur, pool, row)
        if revealed:
            st.markdown(f"**Lab instrument (FTIR) reading of this sample:** cannot rule out `{lab['set']}`, most likely "
                        f"`{BAND_NAME[lab.top_band]}`. This is a model reading of the stored spectrum, not a reference test. "
                        f"Dataset label: {describe(smp)}.")
        pass_status()
        st.caption(
            f"Calibrated on {int(row.n_units):,} {UNIT_WORD[MODE]} · instruments: {SET_LABEL[SENSORS]} · must catch {MUST}% "
            f"or more · {CONF}% sure. " +
            ("Samples come from the 9 locked cells, judged by the model trained on the 30 open cells. " if SCORED else
             "Samples come from the 30 open materials, each judged by a model that never saw that material. ") +
            "Lab-prepared blends, three fillers, 0 to 24% by weight. One reading describes one ground sample, not a container. "
            "ΔE is measured against the pure-cinnamon sample from the same folder when the model may see one, otherwise "
            "against the average of the other folders.")

    with st.expander("How to read this page"):
        st.markdown(
            "1. Press **Next sample**. The screen looks at the photo and the colour reading of one ground sample.\n"
            "2. It gives one of three recommendations:\n"
            f"   - {LANE['PASS']['icon']} **PASS**: the sample looks like pure cinnamon *and* the screen has earned the right to say so. No lab check.\n"
            f"   - {LANE['LAB']['icon']} **LAB**: the screen is not sure. Send the sample for a routine lab check.\n"
            f"   - {LANE['HOLD']['icon']} **HOLD**: the screen is confident the sample has at least {MUST}% filler. Keep the lot back and test it first.\n"
            "3. **Likely filler** and **Filler amount** are the screen's best reading of what was added and how much.\n"
            "4. **Ground truth** is what the dataset says the sample really is, with a check on each thing the screen said. "
            "Switch it off at the top to screen blind.\n"
            "5. Choose your own decision and record it.\n\n"
            "The settings on the left change how cautious the screen is.")

    queue = st.session_state.get("queue", [])
    with st.expander(f"Lab queue this session ({len(queue)})"):
        if not queue:
            st.caption("Nothing has been sent to the lab yet.")
        else:
            cap = st.number_input("Lab capacity (checks per week)", 1, 500, 10)
            q = pd.DataFrame(queue).sort_values(["lane", "p_must_catch"], ascending=[True, False])  # HOLD before LAB
            if len(q) > cap:
                st.error(f"⚠ {len(q)} samples are queued and the lab takes {cap}. The screen is sending more than the lab can check.")
            st.dataframe(q.rename(columns={"p_must_catch": f"P(level ≥ {MUST}%)"}), hide_index=True, width="stretch",
                         column_config={f"P(level ≥ {MUST}%)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    with st.expander("Demo controls"):
        cells = ["Any material"] + sorted(df.cell.unique(), key=lambda c: (c[1], int(c[2:])))
        st.selectbox("Draw the next sample from", cells, key="demo_pick",
                     format_func=lambda c: c if c == "Any material" else cell_name(c),
                     help="For rehearsing the demo. An inspector would not know this.")


# ---------------------------------------------------------------- Evidence
def tab_radar():
    st.info('**In plain words:** the dataset looks like 1,950 separate samples, but many are the same powder measured again and again. This tab shows the proof. It matters because a model tested on near-copies of what it trained on will look far better than it is.')
    rd = R["radar"]
    st.markdown("Each adulterant-and-level **cell** holds 50 sample IDs. They are repeat measurements of one prepared "
                "material, so treating them as 50 independent samples is what field statistics calls *pseudoreplication* "
                "(Hurlbert 1984).")
    a, b, c = st.columns(3)
    a.metric("Sample IDs", f"{R['load_audit']['n_samples']:,}")
    b.metric("Distinct photos", f"{rd['counts'][0]['distinct']:,}")
    c.metric("Independent cells", R["load_audit"]["n_cells"])
    t = pd.DataFrame(rd["counts"])
    t["Strictest count"] = [f"{int(n)} {w}" if isinstance(w, str) else "same as cells" for n, w in zip(t.strict.fillna(0), t.strict_what)]
    t["Distinct measurements"] = [f"{n:,} {w}" for n, w in zip(t.distinct, t.distinct_what)]
    st.dataframe(t.rename(columns={"sensor": "Sensor", "nominal": "Sample IDs", "cells": "Cells"})[
        ["Sensor", "Sample IDs", "Distinct measurements", "Cells", "Strictest count"]], hide_index=True, width="stretch")

    st.subheader("1 · The same photo under several sample IDs")
    tp = rd["photos"]["twin_pair"]
    cols = st.columns([1, 1, 1.4])
    for col, sid in zip(cols, tp["pair"]):
        col.image(str(C.THUMBS / f"{sid}.jpg"), caption=f"{cell_name(sid)} · sample {sid}")
    cols[2].markdown(
        f"These two files differ by **{tp['mean_abs_pixel_difference']:.1f}** grey levels per pixel on average. The same "
        f"photo is stored under **{tp['size']}** sample IDs for {cell_name(tp['group'].split('-')[0])}.\n\n"
        f"Across the dataset the median cell holds {rd['photos']['distinct_per_cell_median']:.0f} distinct photos out of 50, "
        f"and the smallest holds {rd['photos']['distinct_per_cell_min']}.")

    open_cells = sorted(S.cell[~S.locked].unique(), key=lambda c: (c[1], int(c[2:])))
    cell = st.selectbox("Material to inspect", open_cells, index=open_cells.index("CW6"), format_func=cell_name)
    name = cell_name(cell)
    rows = np.flatnonzero((S.cell == cell).to_numpy())

    st.subheader("2 · Fifty spectra from one cell")
    f = rd["ftir"]
    gap = np.full((len(rows), 1), np.nan)  # one trace, with a break between spectra
    fig = go.Figure(go.Scatter(x=np.hstack([np.tile(D.wavenumbers, (len(rows), 1)), gap]).ravel(),
                               y=np.hstack([SPECTRA.raw[rows], gap]).ravel(), mode="lines",
                               line=dict(color="#2a78d6", width=1), opacity=0.3, hoverinfo="skip"))
    chart(fig, 300, title=f"{name}: absorbance of all 50 stored spectra, overlaid",
          xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed"), yaxis=dict(title="Absorbance"))
    groups = "; ".join(f"{C.FOLDER_LABEL[k]}: " + ", ".join("–".join(map(str, (g[0], g[-1]))) + "%" if len(g) > 1 else f"{g[0]}%"
                                                          for g in v["groups"] if len(g) > 1)
                       for k, v in f["blocks"].items() if any(len(g) > 1 for g in v["groups"]))
    st.markdown(
        f"Inside a cell the spectra differ by a standard deviation of **{f['within_cell_sd']:.4f}**, against "
        f"**{f['between_cell_sd']:.4f}** between cells, and the differences are uncorrelated from one wavenumber to the next "
        f"(lag-1 autocorrelation {f['residual_lag1_autocorrelation']:.3f}): one spectrum per cell plus noise.\n\n"
        f"Some cells also share a mean spectrum, leaving **{f['distinct_mean_spectra']}** distinct mean spectra across the "
        f"39 cells. Levels that share one: {groups}.")

    st.subheader("3 · When the cell's photos were taken")
    t = S.shot_time.iloc[rows]
    span = (t.max() - t.min()).total_seconds()
    n = t.value_counts().sort_index()
    fig = go.Figure(go.Scatter(x=n.index, y=[0] * len(n), mode="markers+text", text=[f"{k} IDs" for k in n],
                               textposition="top center", cliponaxis=False,
                               marker=dict(size=8 + 3 * np.sqrt(n.to_numpy()), color="#2a78d6"),
                               hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{text}<extra></extra>"))
    chart(fig, 170, title=f"{name}: {len(n)} distinct photo timestamps for 50 sample IDs, " +
          (f"{span:.0f} seconds apart at most" if span < 3600 else "spread over more than one sitting"),
          yaxis=dict(visible=False, range=[-1, 1.6]), xaxis=dict(tickformat="%H:%M:%S"))
    st.caption(f"Median span across all cells: {rd['photos']['cell_shoot_span_seconds_median']:.0f} seconds. Photo days line up "
               "with folders: " + "; ".join(f"{C.FOLDER_LABEL[k]} {', '.join(v)}" for k, v in rd["photos"]["sessions_by_folder"].items()) + ".")


def tab_ladder():
    st.info('**In plain words:** the same model sits four exams, from easy to fair. In exams 1 and 2 it has already seen near-copies of the test samples. In exam 3 the test material is new, and in exam 4 the whole filler folder is new. Each square is a score (darker is higher). Watch the scores drop from left to right. The bottom-left chart shows error, so there darker is worse.')
    fl = R["floor"]
    st.markdown("One fixed model per sensor set, scored on four splits of rising strictness. Only the split changes.")
    st.caption(f"Floor: always answering “adulterated” scores {pct(fl['always_adulterated_plain_accuracy_all_39_cells'])} plain "
               "accuracy and 50% balanced accuracy, so plain accuracy is never reported. Scores are on the 30 open cells.")
    rows = pd.DataFrame(R["rows"]).query("section == 'ladder'")
    titles = {"detect": "Detect (0% against blend): balanced accuracy", "band": "Band (4 classes): macro-F1",
              "level": "Level: mean absolute error, % points (lower is better)", "type": "Filler type (blends only): macro-F1"}
    cols = st.columns(2) + st.columns(2)
    for col, task in zip(cols, C.TASKS):
        p = rows[(rows.task == task) & (rows.metric == HEADLINE[task])].pivot(index="sensor_set", columns="rung", values="value")
        p = p.reindex(index=list(C.SENSOR_SETS), columns=C.RUNGS)
        text = [["n/a" if np.isnan(v) else (f"{v:.1f}" if task == "level" else f"{v:.2f}") for v in r] for r in p.to_numpy()]
        fig = go.Figure(go.Heatmap(z=p.to_numpy(), x=[C.RUNG_LABEL[r].replace(" · ", "<br>").replace("leave-one-", "leave-") for r in p.columns], y=[SET_LABEL[i] for i in p.index],
                                   text=text, texttemplate="%{text}", colorscale=SEQ, showscale=False, xgap=2, ygap=2,
                                   zmin=0, zmax=None if task == "level" else 1,
                                   hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
        with col:
            chart(fig, 330, title=titles[task], yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom", tickangle=0, title="Split rung"))
    st.caption("Rung 4 is a new-folder test: filler, photo session and cinnamon source change together, so it is not an "
               "unseen-filler result. Filler type has no rung 4 because the held-out filler is never in training. "
               "Rung 3 still leaks for Walnut shell and Wood dust FTIR, where neighbouring levels share one mean spectrum.")


def tab_licence():
    st.info('**In plain words:** PASS lets a sample skip the lab, so the screen has to earn it. This tab shows the two promises the screen must keep, how well each is proven, and how much more evidence it would take.')
    lic, cl = RES.licence, RES.claims
    unit = lic["unit"]
    if lic["licensed"]:
        st.success(f"✓ PASS is licensed: both claims are certified at or above {pct(BAR, 0)}, counting {unit}.")
    else:
        st.error(f"⚠ PASS is locked: counting {unit}, the catch claim is certified at {pct(lic['catch_bound'])} and the "
                 f"reference claim at {pct(lic['reference_bound'])}. Both must reach {pct(BAR, 0)}.")
    names = {"catch": f"1 · Catch: blends at or above {MUST}% are not passed",
             "reference": "2 · Reference: 0% material from a new source is recognised"}
    t = pd.DataFrame([{
        "Claim": names[k],
        "Samples that succeed": f"{v['samples']['k']:,} of {v['samples']['n']:,}",
        "Certified, counting samples": pct(v["samples"]["bound"]),
        "Cells that succeed": f"{v['cells']['k']} of {v['cells']['n']}",
        "Certified, counting cells": pct(v["cells"]["bound"]),
    } for k, v in cl.items()])
    st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Same predictions, two ways of counting. Bounds are one-sided 95% Clopper-Pearson lower bounds on held-out cells: "
               + ("the 30 open cells under leave-one-cell-out, plus the 9 locked cells." if SCORED else
                  "the 30 open cells under leave-one-cell-out.")
               + f" The licence uses the “{unit}” column in this mode. Not a regulatory detection limit.")
    n2 = cl["reference"]["cells"]["n"]
    n_cal = int(RES.df.n_units.iloc[0])
    st.markdown(
        f"**Built-in limits.** The dataset holds {n2} independent 0% materials. Even if all {n2} were recognised, the "
        f"reference claim could be certified at no more than **{pct(cp_lower(n2, n2))}**, so in replicate-aware mode PASS "
        f"stays locked whatever the model does. With {n_cal:,} {UNIT_WORD[MODE]} to calibrate on, α cannot go below "
        f"**{pct(min_alpha(n_cal))}** (so “how sure” tops out at {pct(1 - min_alpha(n_cal), 0)})" + ("; the chosen α is below that, so every band is kept and every sample goes to the lab."
                                          if ALPHA < min_alpha(n_cal) else "."))

    st.subheader("What it would take")
    a, b = st.columns([1, 2])
    target = a.slider("Target to certify", 0.80, 0.995, float(BAR), 0.005, format="%.3f")
    a.metric("Independent confirmed lots needed, zero misses", f"{units_needed(target):,}")
    a.caption("At 95% confidence. " + ", ".join(f"{pct(float(k), 0)}: {v}" for k, v in R["limits"]["lots_needed"].items()) + " lots.")

    held = RES.df[RES.df.level >= MUST]
    ok = (held.assign(p=held.candidate == "PASS").groupby("cell", sort=False).p.mean() <= 0.05).to_numpy()
    with b:
        n = st.slider("Ledger: confirmed held-out cells added", 0, len(ok), len(ok))
        k_cells = np.cumsum(ok)
        k_samples = held.assign(s=held.candidate != "PASS").groupby("cell", sort=False).s.sum().cumsum().to_numpy()
        x = np.arange(1, len(ok) + 1)
        fig = go.Figure()
        for name, y, colr in (("counting samples", [cp_lower(int(k), 50 * i) for i, k in zip(x, k_samples)], "#2a78d6"),
                              ("counting cells", [cp_lower(int(k), int(i)) for i, k in zip(x, k_cells)], "#eb6834")):
            fig.add_trace(go.Scatter(x=x[:n], y=y[:n], mode="lines+markers", name=name, line=dict(color=colr, width=2),
                                     marker=dict(size=8), hovertemplate="%{x} cells: %{y:.1%}<extra>" + name + "</extra>"))
        fig.add_hline(y=BAR, line=dict(color="gray", dash="dot", width=1), annotation_text=f"PASS bar {pct(BAR, 0)}",
                      annotation_position="bottom left")
        chart(fig, 330, title="Catch claim: certified bound as held-out cells are confirmed",
              xaxis=dict(title="Held-out cells confirmed", range=[0.5, len(ok) + 0.5], dtick=2),
              yaxis=dict(range=[0, 1.05], tickformat=".0%"), legend=dict(orientation="h", y=-0.25))
    st.caption("The ledger uses the dataset's own held-out cells, in folder and level order. No lots are simulated.")


def tab_blind_spot():
    st.info('**In plain words:** each square is one material the screen had never seen before. Green means it committed to an answer and was right. Grey means it was not sure and sent the sample to the lab. Red means it was confident and wrong. White squares are set aside for the final test.')
    st.markdown("One tile per adulterant-and-level cell, each judged by a model that never saw it. "
                "**right**: the screen committed and was correct. **LAB**: it abstained. "
                "**wrong**: more than 5% of the cell's samples got a wrong PASS candidate or a HOLD on a 0% sample.")
    t = RES.tiles.set_index(["code", "level"])
    code_of = {"right": 0, "abstains": 1, "wrong": 2}
    z, text, hover = [], [], []
    for code in C.FOLDERS:
        zr, tr, hr = [], [], []
        for level in C.LEVELS:
            if (code, level) in t.index:
                r = t.loc[(code, level)]
                zr.append(code_of[r.status])
                tr.append({"right": "✓ right", "abstains": "⧖ LAB", "wrong": "⚠ wrong"}[r.status])
                hr.append(f"PASS candidates {r.PASS} · LAB {r.LAB} · HOLD {r.HOLD}")
            else:
                zr.append(3)
                tr.append("locked")
                hr.append("Locked until the one-time Sunday scoring")
        z.append(zr)
        text.append(tr)
        hover.append(hr)
    scale = [[0, "#0ca30c"], [0.25, "#0ca30c"], [0.25, "#c3c2b7"], [0.5, "#c3c2b7"], [0.5, "#d03b3b"], [0.75, "#d03b3b"],
             [0.75, "#f0efec"], [1, "#f0efec"]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{l}%" for l in C.LEVELS], y=[C.FOLDER_LABEL[c] for c in C.FOLDERS], text=text,
                               customdata=hover, texttemplate="%{text}", colorscale=scale, zmin=0, zmax=4, showscale=False,
                               xgap=3, ygap=3, hovertemplate="%{y} at %{x}<br>%{customdata}<extra></extra>"))
    chart(fig, 260, title=f"Where the screen is right, abstains or is wrong ({SET_LABEL[SENSORS]}, {MODE_LABEL[MODE].lower()})",
          xaxis=dict(type="category", title="Adulteration level (% w/w)"), yaxis=dict(autorange="reversed"))
    st.caption("PASS candidates are counted before the licence is applied, so the map shows what the model would do if PASS "
               "were allowed. Hover a tile for its counts.")
    with st.expander("Table view"):
        tv = RES.tiles.assign(Filler=RES.tiles.code.map(C.FOLDER_LABEL), Level=RES.tiles.level.astype(str) + "%")
        st.dataframe(tv[["Filler", "Level", "PASS", "LAB", "HOLD", "wrong", "status"]], hide_index=True, width="stretch")


def tab_review():
    st.info('**In plain words:** this compares the screen with an ordinary model that must always give an answer. Saying “not sure, send to lab” costs lab checks, but avoids confident mistakes.')
    m, b, b1 = RES.metrics, RES.baseline, RES.baseline_sample_split
    st.markdown("**B1** is the same model forced to call every sample (0% band means PASS, anything else HOLD). "
                "A wrong auto-decision is a PASS on a blend or a HOLD on a 0% sample.")
    a, c, d = st.columns(3)
    a.metric("Wrong auto-decisions, B1 on held-out cells", f"{b['wrong_auto_decisions']:,}")
    c.metric("Wrong auto-decisions, Earned Pass", f"{m['wrong_auto_decisions']:,}",
             delta=f"{m['wrong_auto_decisions'] - b['wrong_auto_decisions']:+,} against B1", delta_color="inverse")
    d.metric("Samples that end in a lab check", pct(1 - m["pass_share"], 0))
    t = pd.DataFrame({
        "B1 on the sample-ID split": [f"{b1['wrong_auto_decisions']:,} of {b1['auto_decisions']:,}", pct(1.0, 0),
                                      pct(b1["wrong_pass_rate"]), pct(b1["wrong_hold_rate"]), pct(b1["stated_confidence"]),
                                      pct(b1["band_accuracy"]), pct(b1["confidence_gap"])],
        "B1 on held-out cells": [f"{b['wrong_auto_decisions']:,} of {b['auto_decisions']:,}", pct(1.0, 0),
                                 pct(b["wrong_pass_rate"]), pct(b["wrong_hold_rate"]), pct(b["stated_confidence"]),
                                 pct(b["band_accuracy"]), pct(b["confidence_gap"])],
        "Earned Pass on held-out cells": [f"{m['wrong_auto_decisions']:,} of {m['auto_decisions']:,}", pct(m["coverage"]),
                                          pct(m["wrong_pass_rate"]), pct(m["wrong_hold_rate"]), pct(m["stated_coverage"]),
                                          pct(m["set_coverage"]), pct(m["confidence_gap"])],
    }, index=["Wrong auto-decisions", "Coverage (share decided without abstaining)", "Wrong-PASS rate on blends",
              "Wrong-HOLD rate on 0% samples", "Stated confidence", "Observed", "Gap between stated and observed"])
    st.dataframe(t, width="stretch")
    st.caption("Stated and observed: for B1, the mean top probability against band accuracy; for Earned Pass, 1 − α against "
               f"the share of band sets that contain the true band. {m['n_samples']:,} held-out samples in {m['n_cells']} cells.")

    st.subheader("Lab capacity")
    a, c = st.columns(2)
    weekly = a.number_input("Samples screened per week", 1, 5000, 40)
    cap = c.number_input("Lab capacity (checks per week)", 0, 5000, 10)
    need = (1 - m["pass_share"]) * weekly
    if need > cap:
        st.error(f"⚠ Stop rule: the screen would send about {need:.0f} of {weekly} samples to the lab and the lab takes {cap}. "
                 "At these settings the screen has not earned any lab savings.")
    else:
        st.success(f"✓ About {need:.0f} of {weekly} samples would go to the lab, within the capacity of {cap}.")

    st.subheader("Cost panel")
    st.caption("Illustrative assumptions, adjustable. No prices are claimed. Uses the dataset's mix of samples, which is "
               "not a real intake mix.")
    a, c = st.columns(2)
    miss = a.slider("Cost of a wrong PASS, in lab checks", 1, 500, 100)
    delay = c.slider("Extra cost of a wrong HOLD (delay), in lab checks", 0, 20, 2)

    def cost(pass_share, wrong_pass, wrong_hold, blend_share=0.9):
        return 100 * ((1 - pass_share) + miss * wrong_pass * blend_share + delay * wrong_hold * (1 - blend_share))

    y = (S.level.to_numpy()[RES.df.row] > 0).mean()
    p3 = PROBS_BY_SET[SENSORS]
    held = np.where(RES.df.pool.to_numpy()[:, None] == "dev", p3.p3[RES.df.row], p3.pl[RES.df.row])
    b_pass_share = float((held.argmax(1) == 0).mean())
    costs = pd.DataFrame({
        "Strategy": ["Send everything to the lab", "B1: forced calls", "Earned Pass"],
        "Cost per 100 samples, in lab checks": [100.0, cost(b_pass_share, b["wrong_pass_rate"], b["wrong_hold_rate"], y),
                                                cost(m["pass_share"], m["wrong_pass_rate"], m["wrong_hold_rate"], y)],
    })
    st.dataframe(costs, hide_index=True, width="stretch", column_config={
        "Cost per 100 samples, in lab checks": st.column_config.NumberColumn(format="%.0f")})


def tab_controls():
    st.info('**In plain words:** four sanity checks on whether the models learned about the powder itself, or about accidents of how the dataset was made (which day a photo was taken, which folder a file sits in).')
    c = R["controls"]
    st.subheader("Session probe")
    sp = c["session_probe"]
    a, b = st.columns(2)
    a.metric("Photo features predict the shoot day", pct(sp["session"]["balanced_accuracy"]),
             help=f"Balanced accuracy over {sp['session']['classes']} days; chance is {pct(sp['session']['chance'])}.")
    b.metric("Photo features predict the filler", pct(sp["filler"]["balanced_accuracy"]),
             help=f"Balanced accuracy over {sp['filler']['classes']} fillers; chance is {pct(sp['filler']['chance'])}.")
    st.caption(f"Twin-aware split. Chance is {pct(sp['session']['chance'])} for the day and {pct(sp['filler']['chance'])} for "
               "the filler. Photo days line up with folders and levels, so a photo score may be reading the session.")

    st.subheader("Zero-filler probe")
    z = pd.DataFrame(c["zero_filler_probe"]).T
    z.index = [SET_LABEL[i] for i in z.index]
    st.dataframe(z[["own_folder_share"]].rename(columns={"own_folder_share": "0% samples given their own folder's filler"}),
                 width="stretch", column_config={"0% samples given their own folder's filler":
                                                 st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("The filler-type model is trained on blends and applied to the 150 0% samples, which contain no filler. "
               "Chance is 33%. A share well above that means the model reads the folder, not the filler.")

    st.subheader("Offset test")
    o = c["offset_test"]
    t = pd.DataFrame([{"Filler": C.FOLDER_LABEL[k], "Split": C.RUNG_LABEL[r], "Raw absorbance": v["raw"],
                       "First derivative": v["derivative"]} for k, rungs in o.items() for r, v in rungs.items()])
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        k: st.column_config.NumberColumn(format="%.2f") for k in ("Raw absorbance", "First derivative")})
    st.caption("FTIR level error in % points (PLS, 8 components). A first derivative removes a flat offset, so a large rise "
               "means the dose signal was mostly a baseline shift.")

    st.subheader("Band check")
    band = st.selectbox("Wavenumber (cm⁻¹)", list(c["band_check"]))
    fig = go.Figure()
    for code, vals in c["band_check"][band].items():
        fig.add_trace(go.Scatter(x=[int(k) for k in vals], y=list(vals.values()), mode="lines+markers", name=C.FOLDER_LABEL[code],
                                 line=dict(color=FILLER_COLOUR[code], width=2), marker=dict(size=8),
                                 hovertemplate="%{x}%: %{y:.4f}<extra>" + C.FOLDER_LABEL[code] + "</extra>"))
    chart(fig, 330, title=f"Cell-mean absorbance at {band} cm⁻¹ against level (open cells)",
          xaxis=dict(title="Adulteration level (% w/w)"), yaxis=dict(title="Absorbance"), legend=dict(orientation="h", y=-0.3))
    st.caption("FTIR values are assumed to be transmittance and converted with −log10.")


def page_evidence():
    st.title("Evidence")
    st.caption(f"{MODE_LABEL[MODE]} · instruments: {SET_LABEL[SENSORS]} · must catch {MUST}% or more · {CONF}% sure · PASS needs {pct(BAR, 0)} proof. "
               "Change these in the sidebar.")
    tabs = st.tabs(["Replicate Radar", "Stress ladder", "PASS licence", "Blind-spot map", "Review lane vs forced calls", "Controls"])
    for tab, fn in zip(tabs, (tab_radar, tab_ladder, tab_licence, tab_blind_spot, tab_review, tab_controls)):
        with tab:
            fn()


# ---------------------------------------------------------------- Compare
def filler_short(cur, pool, row) -> str:
    if TYPES is None:
        return "not available"
    if row.candidate == "PASS":
        return "none suspected"
    t3, tl = TYPES[SENSORS]
    p = (tl if pool == "locked" else t3)[cur]
    top = int(np.argmax(p))
    if p[top] < FILLER_SURE:
        return "cannot tell which"
    return f"{C.FOLDER_LABEL[list(C.FOLDERS)[top]]} (model {pct(p[top], 0)})" + (", if there is any" if row.in_zero else "")


def reference_delta(cur, pool) -> float:
    """Colour distance to the 0% reference the model may see for this sample."""
    smp, dev = S.iloc[cur], D.dev
    train = dev[S.cell.to_numpy()[dev] != smp.cell] if pool == "dev" else dev
    refs = color.references(S, train)
    return float(np.linalg.norm(smp[["L", "a", "b"]].to_numpy(float) - refs.get(smp.code, refs[None])))


def page_compare():
    st.title("Compare two samples")
    st.caption("Pick two samples to see their screening results side by side. Both use the settings on the left. "
               "Unlike Inspect, this page shows what each sample really is from the start.")
    pool = "locked" if SCORED else "dev"
    df = RES.df[RES.df.pool == pool].set_index("row")
    cells = sorted(df.cell.unique(), key=lambda c: (c[1], int(c[2:])))
    defaults = {"A": cells[0], "B": cells[len(cells) // 2]}
    for side in "AB":
        if st.session_state.get(f"cmp_cell_{side}") not in cells:
            st.session_state[f"cmp_cell_{side}"] = defaults[side]

    def from_inspect():
        smp = S.iloc[st.session_state.cur]
        st.session_state.cmp_cell_A, st.session_state.cmp_rep_A = smp.cell, int(smp.rep)

    if st.session_state.get("cur") in df.index:
        st.button("Use the sample from Inspect as sample A", on_click=from_inspect)

    picked = {}
    cols = st.columns(2, gap="large")
    for side, col in zip("AB", cols):
        with col:
            st.subheader(f"Sample {side}")
            a, b = st.columns([2, 1])
            cell = a.selectbox("Material", cells, format_func=cell_name, key=f"cmp_cell_{side}")
            rep_no = b.number_input("Sample number", 1, 50, 1, key=f"cmp_rep_{side}",
                                    help="Each material has 50 samples, numbered 1 to 50.")
            cur = int(np.flatnonzero(((S.cell == cell) & (S.rep == rep_no)).to_numpy())[0])
            row, smp = df.loc[cur], S.iloc[cur]
            picked[side] = (cur, row, smp)
            st.image(str(C.THUMBS / f"{smp.sample_id}.jpg"), caption=describe(smp))
            lane_card(row.lane, decide.reason(row, RES))

    (ca, ra, sa), (cb, rb, sb) = picked["A"], picked["B"]
    st.divider()
    if ca == cb:
        st.info("Both sides show the same sample. Pick a different material or sample number on one side.")
    elif ra.lane == rb.lane:
        st.success(f"Both samples get the same recommendation: **{ra.lane}**.")
    else:
        st.warning(f"The recommendations differ: sample A gets **{ra.lane}**, sample B gets **{rb.lane}**.")

    def shortlist(row):
        return ", ".join(BAND_NAME[b] for b in C.BANDS if row[f"in_{b}"]) or "none fit"

    def column(cur, row, smp):
        return [
            "pure cinnamon (0%)" if smp.level == 0 else f"{C.FOLDER_LABEL[smp.code]}, {smp.level}%",
            f"{LANE[row.lane]['icon']} {row.lane}",
            shortlist(row),
            "yes" if row.covered else "no",
            filler_short(cur, pool, row),
            estimate_text(cur, pool),
            f"{smp.L:.1f}", f"{smp.a:.1f}", f"{smp.b:.1f}",
            f"{reference_delta(cur, pool):.1f}",
        ]

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Side by side")
        table = pd.DataFrame({"Sample A": column(ca, ra, sa), "Sample B": column(cb, rb, sb)}, index=[
            "What it really is", "Recommendation", "Filler levels not ruled out", "True level is on that shortlist",
            "Likely filler", "Estimated filler amount", "L* (lightness)", "a* (green to red)", "b* (blue to yellow)", "ΔE to its reference"])
        st.dataframe(table, width="stretch")
        gap = float(np.linalg.norm(sa[["L", "a", "b"]].to_numpy(float) - sb[["L", "a", "b"]].to_numpy(float)))
        st.metric("Colour difference between A and B (ΔE)", f"{gap:.1f}",
                  help="Distance between the two colorimeter readings. Under about 2 is hard to see by eye.")
    with right:
        fig = go.Figure()
        for name, row, colr in (("Sample A", ra, "#2a78d6"), ("Sample B", rb, "#eb6834")):
            fig.add_trace(go.Bar(y=[BAND_NAME[b] for b in C.BANDS], x=[row[f"p_{b}"] for b in C.BANDS], orientation="h",
                                 name=name, marker_color=colr, text=[pct(row[f"p_{b}"], 0) for b in C.BANDS],
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x:.1%}<extra>" + name + "</extra>"))
        chart(fig, 360, title="How likely each filler level is, according to the model", barmode="group", bargap=0.3,
              xaxis=dict(range=[0, 1.12], tickformat=".0%", showgrid=False), yaxis=dict(autorange="reversed"),
              legend=dict(orientation="h", y=-0.15, traceorder="normal"))
    st.caption("The model is often very sure and still wrong on materials it has not seen, which is why the shortlist "
               "can be longer than the bars suggest.")


# ---------------------------------------------------------------- Log
def page_log():
    st.title("Decision log")
    st.caption("One row per recorded decision. The inspector answers for the record; the tool only recommends.")
    if not LOG.exists():
        st.info("No decisions recorded yet. Record one on the Inspect page.")
        return
    log = pd.read_csv(LOG)
    a, b, c = st.columns(3)
    a.metric("Decisions", len(log))
    b.metric("Overrides", int(log.override.sum()))
    c.metric("Sent to lab", int(log.sent_to_lab.sum()))
    log.insert(2, "material", log.sample_id.map(cell_name))
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
    st.download_button("Download CSV", LOG.read_bytes(), "earned_pass_decision_log.csv", "text/csv")


st.navigation([
    st.Page(page_inspect, title="Inspect", icon=":material/search:", url_path="inspect", default=True),
    st.Page(page_compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(page_evidence, title="Evidence", icon=":material/fact_check:", url_path="evidence"),
    st.Page(page_log, title="Log", icon=":material/list_alt:", url_path="log"),
]).run()
