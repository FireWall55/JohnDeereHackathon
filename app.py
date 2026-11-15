"""Earned Pass: a receiving screen for ground cinnamon.

    streamlit run app.py

Every sample is compared with its supplier's own reference sample. The screen
estimates the chance each filler is present and applies the team's verdict
rules (earnedpass/verdict.py): reject wood dust, send possible wood to the lab,
label peanut and walnut shell, and send everything else to the lab to verify
it is pure. Reads artifacts written by `python -m earnedpass.run_all`.
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
from earnedpass import models
from earnedpass import verdict as V
from earnedpass.eval.ladder import HEADLINE
from earnedpass.run_all import PROBS, PROBS_REF, RESULTS

LOG = C.ARTIFACTS / "decision_log.csv"
LOG_COLUMNS = ["time", "sample_id", "model_version", "sensors", "wood_lab", "wood_reject", "peanut_limit",
               "walnut_limit", "recommended", "labels", "decision", "override", "sent_to_lab", "user"]

# Status colours always travel with an icon and a word, never alone.
LANE = {
    V.REJECT: {"title": "REJECT", "icon": "✕", "colour": "#d03b3b", "text": "wood dust: reject and discard the lot"},
    V.LAB_WOOD: {"title": "LAB", "icon": "⚠", "colour": "#ec835a", "text": "possible wood dust: confirm at the lab before any use"},
    V.LABEL: {"title": "LABEL", "icon": "🏷", "colour": "#2a78d6", "text": "nut-shell material found: label the lot"},
    V.LAB_PURE: {"title": "LAB", "icon": "⧖", "colour": "#fab219", "text": "no filler found: the lab verifies it is pure"},
}
LANE_CHOICE = {V.REJECT: "✕ Reject", V.LAB_WOOD: "⚠ Lab (wood risk)", V.LABEL: "🏷 Label", V.LAB_PURE: "⧖ Lab (verify pure)"}
FILLER_COLOUR = {"P": "#2a78d6", "W": "#eb6834", "D": "#1baf7a"}
SEQ = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
SET_LABEL = {"colour": "Colorimeter", "ftir": "FTIR", "image": "Photo", "colour+ftir": "Colorimeter + FTIR",
             "colour+image": "Colorimeter + photo", "ftir+image": "FTIR + photo", "all": "All three"}
TRUTH = {"P": "Peanut shell", "W": "Walnut shell", "D": "Wood dust"}

st.set_page_config(page_title="Earned Pass", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------- loading
def _read(path):
    z = np.load(path)
    out = {}
    for i, ss in enumerate(z["sets"]):
        out[ss] = {"band": (z["p3"][i], z["pl"][i]), "types": (z["t3"][i], z["tl"][i]),
                   "levels": (z["e3"][i], z["el"][i])}
    return out, (z["is_ref"] if "is_ref" in z else None)


@st.cache_resource(show_spinner="Loading artifacts...")
def load(_stamp):
    d = data_mod.get()
    results = json.loads(RESULTS.read_text())
    general, _ = _read(PROBS)
    supplier, is_ref = _read(PROBS_REF)
    return d, results, {"general": general, "supplier": supplier}, is_ref, models.Spectra(d.X_ftir)


@st.cache_data(show_spinner=False)
def verdicts(_stamp, ref, sensors, th: V.Thresholds) -> pd.DataFrame:
    d, _, art, _, _ = load(_stamp)
    a = art[ref][sensors]
    return V.screened(d.samples, a["band"], a["types"], th)


if not (RESULTS.exists() and PROBS.exists() and PROBS_REF.exists()):
    st.error("No results yet. Build them first:")
    st.code("python -m earnedpass.run_all", language="bash")
    st.stop()

STAMP = RESULTS.stat().st_mtime
D, R, ART, IS_REF, SPECTRA = load(STAMP)
S = D.samples
SCORED = bool(R["meta"]["locked_scored_at"])
POOL = "locked" if SCORED else "dev"

with st.sidebar:
    st.header("Screen settings")
    st.caption("A QC manager sets these once, before the screen is used. The starting values are the team's defaults.")
    EXPLAIN = st.toggle("Explain each setting", True)

    def note(text):
        if EXPLAIN:
            st.caption(text)

    st.caption("**Compared with:** each supplier's own reference sample. ⚠️ The pure samples tested come from the "
               "same batch as the reference, so results on pure cinnamon are optimistic.")
    SENSORS = st.selectbox("Instruments used for screening", list(C.SENSOR_SETS), index=list(C.SENSOR_SETS).index(C.SCREEN_SET),
                           format_func=SET_LABEL.get)
    note("A colorimeter and a camera are what an inspector has at the receiving dock. FTIR is a lab instrument.")

    st.subheader("Verdict limits")
    note("Each limit is compared with the **chance that filler is present**: the chance of any filler × the chance "
         "it is that filler. Lower a limit to be stricter.")
    WOOD_LAB = st.slider("Wood dust: send to lab above", 1, 20, 1, 1, format="%d%%")
    note("Any sign of wood dust goes to the lab before the lot is used.")
    WOOD_REJECT = st.slider("Wood dust: reject above", WOOD_LAB, 50, max(10, WOOD_LAB), 1, format="%d%%")
    note("Above this the lot is rejected and discarded. Wood is never accepted.")
    PEANUT = st.slider("Peanut shell: label above", 1, 50, 10, 1, format="%d%%")
    WALNUT = st.slider("Walnut shell: label above", 1, 50, 10, 1, format="%d%%")
    note("Above these the lot is labelled as containing that nut shell. Both can be labelled at once; the screen "
         "does not choose between them. Below every limit, the lot goes to the lab to verify it is pure.")

    USER = st.text_input("Your name", "inspector-1")
    note("Written into the decision log next to every decision you record.")
    st.divider()
    st.caption("The tool recommends; the inspector decides. It supplements lab testing, it does not replace it.")
    st.caption(f"Model {R['meta']['model_version']} · " + ("locked cells scored " + R["meta"]["locked_scored_at"] if SCORED
               else "locked cells (levels 3, 10, 18) not scored yet"))

TH = V.Thresholds(WOOD_LAB / 100, WOOD_REJECT / 100, PEANUT / 100, WALNUT / 100)
VERD = verdicts(STAMP, "supplier", SENSORS, TH)


# ---------------------------------------------------------------- helpers
def pct(x, digits=1):
    return f"{x:.{digits}%}"


def chart(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **kw)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def cell_name(cell: str) -> str:
    """'CW6' -> 'Walnut shell, 6%'. Also accepts a full sample ID such as 'CW6S24'."""
    code, level = cell[1], cell[2:].split("S")[0]
    return f"0% ({C.FOLDER_LABEL[code]} folder)" if level == "0" else f"{C.FOLDER_LABEL[code]}, {level}%"


def describe(sample) -> str:
    if sample.level == 0:
        return f"{sample.sample_id}: 0%, the unblended material in the {C.FOLDER_LABEL[sample.code]} folder"
    return f"{sample.sample_id}: {C.FOLDER_LABEL[sample.code]}, {sample.level}% w/w"


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



def lane_card(row):
    m = LANE[row.lane]
    labels = label_text(row)
    extra = f"<div style='margin-top:.3rem'><b>Label:</b> {labels}</div>" if labels else ""
    st.markdown(
        f"""<div style="border-left:6px solid {m['colour']};padding:.6rem 1rem;border-radius:6px;
        background:color-mix(in srgb,{m['colour']} 9%,transparent)">
        <div style="font-size:1.9rem;font-weight:700;line-height:1.2">{m['icon']} {m['title']}</div>
        <div style="opacity:.8">{m['text']}</div>{extra}</div>""",
        unsafe_allow_html=True,
    )


def label_text(row) -> str:
    names = [n for n, on in (("contains peanut shell", row.label_peanut), ("contains walnut shell", row.label_walnut)) if on]
    return " and ".join(names)


def reason(row) -> str:
    if row.lane == V.REJECT:
        s = f"The chance of wood dust is {pct(row.p_wood, 0)}, above the {WOOD_REJECT}% reject limit."
    elif row.lane == V.LAB_WOOD:
        s = (f"The chance of wood dust is {pct(row.p_wood, 0)}: above the {WOOD_LAB}% lab limit, below the "
             f"{WOOD_REJECT}% reject limit. Confirm at the lab before any use.")
    elif row.lane == V.LABEL:
        s = "Nut-shell material is likely: " + ", ".join(
            f"{n} {pct(p, 0)} (limit {lim}%)" for n, p, lim, on in
            (("peanut shell", row.p_peanut, PEANUT, row.label_peanut), ("walnut shell", row.p_walnut, WALNUT, row.label_walnut))
            if on) + ". Wood dust is below its limits."
    else:
        s = ("No filler is above its limit, so the lot goes to the lab to verify it is pure. "
             f"Chance of any filler: {pct(row.p_any, 0)}.")
    if row.lane in (V.REJECT, V.LAB_WOOD) and (row.label_peanut or row.label_walnut):
        s += f" Nut-shell material is also likely ({label_text(row)})."
    return s


def chance_rows(row) -> dict:
    def mark(p, lim):
        return f"{pct(p, 0)} · {'above' if p > lim / 100 else 'below'} the {lim}% label limit"
    if row.p_wood > WOOD_REJECT / 100:
        wood = f"{pct(row.p_wood, 0)} · ✕ above the {WOOD_REJECT}% reject limit"
    elif row.p_wood > WOOD_LAB / 100:
        wood = f"{pct(row.p_wood, 0)} · ⚠ above the {WOOD_LAB}% lab limit"
    else:
        wood = f"{pct(row.p_wood, 0)} · below the {WOOD_LAB}% lab limit"
    return {"Chance of peanut shell": mark(row.p_peanut, PEANUT), "Chance of walnut shell": mark(row.p_walnut, WALNUT),
            "Chance of wood dust": wood}


def level_estimate(cur):
    """(estimate in %, margin in points). The margin covers 8 in 10 held-out samples."""
    e3, el = ART["supplier"][SENSORS]["levels"]
    est = float((el if POOL == "locked" else e3)[cur])
    dev = D.dev[np.isfinite(e3[D.dev])]
    margin = float(np.quantile(np.abs(e3[dev] - S.level.to_numpy()[dev]), 0.8))
    return est, margin


def estimate_text(cur) -> str:
    est, m = level_estimate(cur)
    return f"about {est:.0f}%  (likely {max(0, est - m):.0f}–{min(24, est + m):.0f}%)"


def reference_delta(cur) -> float:
    smp = S.iloc[cur]
    ref = S[IS_REF & (S.code == smp.code).to_numpy()][["L", "a", "b"]].mean().to_numpy()
    return float(np.linalg.norm(smp[["L", "a", "b"]].to_numpy(float) - ref))


def truth_name(smp) -> str:
    return "pure cinnamon (0%)" if smp.level == 0 else f"{TRUTH[smp.code]}, {smp.level}%"


def verdict_check(row, smp) -> str:
    if smp.level == 0:
        return "✓ right: pure, sent to verify" if row.lane == V.LAB_PURE else (
            "✗ wrong: pure cinnamon rejected" if row.lane == V.REJECT else "✗ wrong: pure cinnamon flagged")
    if smp.code == "D":
        return "✓ right: wood caught" if row.lane in (V.REJECT, V.LAB_WOOD) else "✗ wrong: wood dust let through"
    if row.lane in (V.REJECT, V.LAB_WOOD):
        return "✗ wrong: flagged as wood"
    if row.lane == V.LAB_PURE:
        return "✗ wrong: filler missed (sent to verify pure)"
    return {"right": "✓ right: correct label", "partly": "◐ partly: correct label plus the other nut",
            "wrong": "✗ wrong: wrong nut labelled"}[row.result]


TINT = {"green": "rgba(12,163,12,.22)", "red": "rgba(208,59,59,.28)", "orange": "rgba(236,131,90,.30)",
        "blue": "rgba(42,120,214,.26)", "yellow": "rgba(250,178,25,.28)"}


def chance_colour(p, limits):
    """limits: [(limit %, colour)] from lowest to highest; green when below them all."""
    colour = "green"
    for lim, c in limits:
        if p > lim / 100:
            colour = c
    return colour


def check_colour(text: str):
    return "green" if text.startswith("✓") else ("yellow" if text.startswith("◐") else "red")


def sample_rows(row, smp, cur, show_truth, extras=False):
    """(rows, colours) for the sample table. Colours: green fine, red reject/wrong, orange wood risk, blue label."""
    rows, colours = chance_rows(row), {}
    colours["Chance of peanut shell"] = chance_colour(row.p_peanut, [(PEANUT, "blue")])
    colours["Chance of walnut shell"] = chance_colour(row.p_walnut, [(WALNUT, "blue")])
    colours["Chance of wood dust"] = chance_colour(row.p_wood, [(WOOD_LAB, "orange"), (WOOD_REJECT, "red")])
    rows["Estimated filler amount"] = estimate_text(cur)
    if show_truth:
        est, m = level_estimate(cur)
        off = abs(est - smp.level)
        rows["Ground truth"] = truth_name(smp)
        rows["Verdict was"] = verdict_check(row, smp)
        rows["Amount estimate was"] = f"{'✓' if off <= m else '✗'} off by {off:.0f} points"
        colours["Verdict was"] = check_colour(rows["Verdict was"])
        colours["Amount estimate was"] = check_colour(rows["Amount estimate was"])
    else:
        rows["Ground truth"] = "hidden until you record a decision"
    if extras:
        rows["Chance of any filler"] = pct(row.p_any, 0)
        rows["Compared with"] = f"the {TRUTH[smp.code].lower()} supplier's reference sample"
        rows.update({"L* (lightness)": f"{smp.L:.1f}", "a* (green to red)": f"{smp.a:.1f}",
                     "b* (blue to yellow)": f"{smp.b:.1f}", "ΔE to the reference": f"{reference_delta(cur):.1f}"})
    return rows, colours


def styled(columns: dict):
    """columns: {name: (rows, colours)} -> a colour-coded table."""
    df = pd.DataFrame({name: rc[0] for name, rc in columns.items()})

    def paint(col):
        colours = columns[col.name][1]
        return [f"background-color: {TINT[colours[k]]}" if k in colours else "" for k in col.index]

    return df.style.apply(paint, axis=0)


BANNER = {V.REJECT: ("#d03b3b", "#ffffff"), V.LAB_WOOD: ("#ec835a", "#2c1206"), V.LABEL: ("#2a78d6", "#ffffff"),
          V.LAB_PURE: ("#fab219", "#2c2006")}


def instruction(row) -> str:
    if row.lane == V.REJECT:
        return "Discard this lot."
    if row.lane == V.LAB_WOOD:
        return "Send to the lab before any use."
    if row.lane == V.LABEL:
        return f"Label as: {label_text(row)}."
    return "Send to the lab to confirm it is pure."


def verdict_banner(row):
    bg, fg = BANNER[row.lane]
    m = LANE[row.lane]
    st.markdown(
        f"""<div style="background:{bg};color:{fg};padding:.9rem 1.2rem;border-radius:10px">
        <div style="font-size:.85rem;font-weight:600;opacity:.85">① THE VERDICT</div>
        <div style="font-size:2.1rem;font-weight:800;line-height:1.15">{m['icon']} {m['title']}</div>
        <div style="font-size:1.15rem;font-weight:600">{instruction(row)}</div></div>""",
        unsafe_allow_html=True)


def truth_badge(row, smp, show):
    if not show:
        bg, fg, head, body = "#888780", "#ffffff", "? HIDDEN", "Ground truth appears after you record a decision."
    else:
        check = verdict_check(row, smp)
        kind = check_colour(check)
        bg, fg, head = {"green": ("#0ca30c", "#ffffff", "✓ RIGHT"), "red": ("#d03b3b", "#ffffff", "✗ WRONG"),
                        "yellow": ("#fab219", "#2c2006", "◐ PARTLY")}[kind]
        body = f"It is {truth_name(smp)}." + (f" {check.split(': ', 1)[1].capitalize()}." if ": " in check else "")
    st.markdown(
        f"""<div style="background:{bg};color:{fg};padding:.9rem 1.2rem;border-radius:10px;height:100%">
        <div style="font-size:.85rem;font-weight:600;opacity:.85">WAS THE SCREEN RIGHT?</div>
        <div style="font-size:2.1rem;font-weight:800;line-height:1.15">{head}</div>
        <div style="font-size:1rem">{body}</div></div>""",
        unsafe_allow_html=True)


def append_log(row: dict):
    if LOG.exists() and pd.read_csv(LOG, nrows=0).columns.tolist() != LOG_COLUMNS:
        LOG.rename(LOG.with_name("decision_log_old.csv"))  # older format: kept, not mixed in
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


ZONE_COLOUR = {"below": "#9c9a92", "label": "#2a78d6", "lab": "#ec835a", "reject": "#d03b3b"}


def filler_meters(row):
    """Three bars, one per filler, each with its limit(s) drawn as lines."""
    items = [("Peanut shell", row.p_peanut, [(PEANUT, "label")]),
             ("Walnut shell", row.p_walnut, [(WALNUT, "label")]),
             ("Wood dust", row.p_wood, [(WOOD_LAB, "lab"), (WOOD_REJECT, "reject")])]
    names, vals, colours, words = [], [], [], []
    for name, v, limits in items:
        zone, word = "below", "below its limit"
        for lim, z in limits:
            if v > lim / 100:
                zone = z
                word = {"label": "label it", "lab": "send to lab", "reject": "reject"}[z]
        names.append(name)
        vals.append(v)
        colours.append(ZONE_COLOUR[zone])
        words.append(f"{pct(v, 0)} · {word}")
    fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker_color=colours, width=0.55, text=words,
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    for i, (_, _, limits) in enumerate(items):
        for lim, z in limits:
            fig.add_shape(type="line", x0=lim / 100, x1=lim / 100, y0=i - 0.42, y1=i + 0.42,
                          line=dict(color=ZONE_COLOUR[z], width=2, dash="dot"))
    chart(fig, 190, title="Chance each filler is present (dotted lines are the limits)",
          xaxis=dict(range=[0, 1.35], tickformat=".0%", showgrid=False, tickvals=[0, 0.25, 0.5, 0.75, 1]),
          yaxis=dict(autorange="reversed"), showlegend=False)


def lab_to_hex(L, a, b) -> str:
    """CIE L*a*b* (D65) to an sRGB hex colour, for a swatch."""
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    def finv(f):
        return f ** 3 if f ** 3 > 216 / 24389 else (116 * f - 16) / (24389 / 27)

    x, y, z = 0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)
    rgb = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252]]) @ np.array([x, y, z])
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    r, g, bl = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{bl:02x}"


def reference_lab(code):
    return S[IS_REF & (S.code == code).to_numpy()][["L", "a", "b"]].mean().to_numpy()


def swatches(smp, cur):
    ref = reference_lab(smp.code)
    box = ("display:inline-block;width:64px;height:40px;border-radius:6px;border:1px solid rgba(128,128,128,.4);"
           "vertical-align:middle")
    d = reference_delta(cur)
    look = "hard to see by eye" if d < 2 else ("visible side by side" if d < 5 else "clearly different")
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <div style="text-align:center"><span style="{box};background:{lab_to_hex(*ref)}"></span>
        <div style="font-size:.8rem;opacity:.75">Supplier's reference</div></div>
        <div style="text-align:center;font-size:.9rem">ΔE {d:.1f}<br><span style="opacity:.75">{look}</span></div>
        <div style="text-align:center"><span style="{box};background:{lab_to_hex(smp.L, smp.a, smp.b)}"></span>
        <div style="font-size:.8rem;opacity:.75">This sample</div></div></div>""",
        unsafe_allow_html=True)
    st.caption("Colours as measured by the colorimeter. A screen shows them only approximately.")


def decision_path(row):
    """Which rule decided the verdict, as a row of steps."""
    steps = [("Wood above reject limit?", row.lane == V.REJECT),
             ("Wood above lab limit?", row.lane == V.LAB_WOOD),
             ("Peanut or walnut above label limit?", row.lane == V.LABEL),
             ("Nothing above a limit", row.lane == V.LAB_PURE)]
    fired = [i for i, (_, on) in enumerate(steps) if on][0]
    pills = []
    for i, (text, on) in enumerate(steps):
        if i < fired:
            style, mark = "opacity:.55", "no"
        elif i == fired:
            style, mark = f"border:2px solid {LANE[row.lane]['colour']};font-weight:600", LANE_CHOICE[row.lane]
        else:
            style, mark = "opacity:.3", "not checked"
        pills.append(f'<div style="padding:.35rem .6rem;border-radius:8px;border:1px solid rgba(128,128,128,.4);{style}">'
                     f'{text}<br><span style="font-size:.8rem">{mark}</span></div>')
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:stretch;font-size:.85rem">'
                + '<div style="align-self:center">→</div>'.join(pills) + "</div>", unsafe_allow_html=True)


def how_far_to_trust():
    t3 = ART["supplier"][SENSORS]["types"][0]
    blends = D.dev[(S.level.to_numpy()[D.dev] > 0) & np.isfinite(t3[D.dev]).all(1)]
    hit = float((t3[blends].argmax(1) == S.code.iloc[blends].map(list(C.FOLDERS).index).to_numpy()).mean())
    st.markdown(
        f"- The screen only knows three fillers: peanut shell, walnut shell and wood dust. A filler it has never seen "
        "would be given one of these names.\n"
        f"- On blends it had not seen, these instruments named the right filler for **{pct(hit)}** of samples.\n"
        "- In this dataset each filler was prepared and photographed in its own batch, on its own days. The result may "
        "partly be recognising the batch, not the filler.\n"
        "- The pure samples tested come from the same batch as the supplier's reference, so results on pure cinnamon "
        "are optimistic.\n"
        "- It measures filler, not allergens. A nut-shell label means nut-shell material is likely, not that allergen "
        "content was tested.")


# ---------------------------------------------------------------- Inspect
def page_inspect():
    st.title("Sample screening")
    df = VERD[VERD.pool == POOL].set_index("row")
    pick = st.session_state.get("demo_pick", "Any material")
    ss = st.session_state
    ss.setdefault("hist", [])
    ss.setdefault("pos", -1)
    ss.setdefault("recorded", {})
    ss.hist = [h for h in ss.hist if h in df.index]
    ss.pos = min(ss.pos, len(ss.hist) - 1)

    def go_next():
        if ss.pos < len(ss.hist) - 1:
            ss.pos += 1
        else:
            choices = df.index[df.cell == pick] if pick in set(df.cell) else df.index
            ss.hist.append(int(np.random.default_rng().choice(choices)))
            ss.pos = len(ss.hist) - 1

    def go_prev():
        ss.pos = max(0, ss.pos - 1)

    if ss.pos < 0:
        go_next()
    cur = ss.hist[ss.pos]
    ss.cur = cur
    row, smp = df.loc[cur], S.iloc[cur]
    revealed = cur in ss.recorded

    a, b, c, d = st.columns([0.8, 0.8, 1.2, 2.2], vertical_alignment="center")
    a.button("◀ Previous", on_click=go_prev, disabled=ss.pos == 0, width="stretch")
    b.button("Next ▶", on_click=go_next, type="primary", width="stretch")
    c.markdown(f"Sample **{ss.pos + 1}** of **{len(ss.hist)}** viewed")
    show_truth = d.toggle("Show ground truth", True, help="What the dataset says the sample really is, and whether the "
                          "verdict was right. Turn off to screen blind; it then appears after you record a decision.")
    truth_visible = show_truth or revealed

    a, b = st.columns([1.6, 1], gap="medium")
    with a:
        verdict_banner(row)
    with b:
        truth_badge(row, smp, truth_visible)

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.image(str(C.THUMBS / f"{smp.sample_id}.jpg"), caption=f"Sample #{cur:04d}")
    with right:
        st.markdown("**② Why:** the chance of each filler, against the limits in the sidebar")
        filler_meters(row)
        st.dataframe(styled({"This sample": sample_rows(row, smp, cur, truth_visible)}), width="stretch")
        st.caption("Green: below its limit, or right. Blue: label. Orange: wood risk. Red: reject, or wrong.")

    a, b = st.columns([1.3, 1], gap="large", vertical_alignment="bottom")
    decision = a.radio("Your decision", V.LANES, index=V.LANES.index(ss.recorded.get(cur, row.lane)), horizontal=True,
                       key=f"decision{cur}", format_func=LANE_CHOICE.get, disabled=revealed)
    if b.button("Record decision", disabled=revealed):
        append_log(dict(time=dt.datetime.now().isoformat(timespec="seconds"), sample_id=smp.sample_id,
                        model_version=R["meta"]["model_version"], sensors=SENSORS, wood_lab=WOOD_LAB,
                        wood_reject=WOOD_REJECT, peanut_limit=PEANUT, walnut_limit=WALNUT, recommended=row.lane,
                        labels=label_text(row), decision=decision, override=decision != row.lane,
                        sent_to_lab=decision in (V.LAB_WOOD, V.LAB_PURE), user=USER))
        if decision in (V.LAB_WOOD, V.LAB_PURE):
            ss.setdefault("queue", []).append(
                dict(sample=f"#{cur:04d}", reason=LANE[decision]["text"], chance_of_filler=float(row.p_any)))
        ss.recorded[cur] = decision
        st.rerun()
    if decision != row.lane and not revealed:
        st.warning(f"This overrides the recommendation ({LANE_CHOICE[row.lane]}). The override is recorded under your name.")
    if revealed:
        st.success(f"Decision recorded for this sample: {LANE_CHOICE[ss.recorded[cur]]}.")

    with st.expander("③ Details and reasoning"):
        st.markdown(f"**Why {LANE[row.lane]['title']}:** {reason(row)}")
        st.markdown("**How the verdict was reached**")
        decision_path(row)
        st.markdown(f"**Compared with** the {TRUTH[smp.code].lower()} supplier's reference sample")
        swatches(smp, cur)
        st.markdown(f"Colour reading: L\\* {smp.L:.1f} (lightness) · a\\* {smp.a:.1f} (green to red) · "
                    f"b\\* {smp.b:.1f} (blue to yellow) · ΔE to the reference {reference_delta(cur):.1f}")
        t3, tl = ART["supplier"][SENSORS]["types"]
        p = (tl if POOL == "locked" else t3)[cur]
        st.markdown(f"**How the chances are worked out.** Chance of any filler (from the amount-band model): "
                    f"**{pct(row.p_any, 0)}**. If there is filler, which one (from the filler model): peanut shell "
                    f"{pct(p[V.P], 0)}, walnut shell {pct(p[V.W], 0)}, wood dust {pct(p[V.D], 0)}. Each chance in the "
                    "table is the two multiplied.")
        est, m = level_estimate(cur)
        st.caption(f"The estimated amount comes from a separate model. On materials it had not seen, it was within "
                   f"±{m:.0f} points of the truth for 8 samples in 10.")
        st.markdown("**How far to trust it**")
        how_far_to_trust()

    with st.expander("How to read this page"):
        st.markdown(
            "1. Press **Next ▶** for a new sample, **◀ Previous** to go back. The screen compares the photo and colour reading with the supplier's reference.\n"
            "2. It works out the chance that each filler is present and gives one verdict:\n"
            f"   - {LANE_CHOICE[V.REJECT]}: wood dust above the reject limit. Discard the lot.\n"
            f"   - {LANE_CHOICE[V.LAB_WOOD]}: wood dust above the lab limit. Confirm before any use.\n"
            f"   - {LANE_CHOICE[V.LABEL]}: peanut or walnut shell above its limit. Label the lot; both can apply.\n"
            f"   - {LANE_CHOICE[V.LAB_PURE]}: nothing above its limit. The lab verifies it is pure.\n"
            "3. The limits are in the sidebar.\n"
            "4. **Ground truth** shows what the sample really is and whether the verdict was right.\n"
            "5. Choose your own decision and record it.")

    queue = st.session_state.get("queue", [])
    with st.expander(f"Lab queue this session ({len(queue)})"):
        if not queue:
            st.caption("Nothing has been sent to the lab yet.")
        else:
            cap = st.number_input("Lab capacity (checks per week)", 1, 500, 10)
            q = pd.DataFrame(queue).sort_values("chance_of_filler", ascending=False)
            if len(q) > cap:
                st.error(f"⚠ {len(q)} samples are queued and the lab takes {cap}.")
            st.dataframe(q, hide_index=True, width="stretch", column_config={"chance_of_filler": st.column_config.ProgressColumn(
                "Chance of any filler", format="percent", min_value=0, max_value=1)})

    with st.expander("Demo controls"):
        cells = ["Any material"] + sorted(df.cell.unique(), key=lambda c: (c[1], int(c[2:])))
        st.selectbox("Draw the next sample from", cells, key="demo_pick",
                     format_func=lambda c: c if c == "Any material" else cell_name(c),
                     help="For rehearsing the demo. An inspector would not know this.")


# ---------------------------------------------------------------- How it works
WORKFLOW_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.32;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#9c9a92",
        fillcolor="#f1efe8", fontcolor="#2c2c2a", margin="0.12,0.06"];
  edge [color="#9c9a92", fontname="Helvetica", fontsize=9, fontcolor="#888780"];
  lot [label="1. Lot arrives\nfrom a supplier"];
  sample [label="2. Take a sample\nphoto + colour reading"];
  ref [label="3. Compare with that\nsupplier's reference"];
  models [label="4. Models estimate the\nchance of each filler"];
  rules [label="5. Verdict rules\n(limits in the sidebar)"];
  reject [label="Reject\ndiscard the lot", fillcolor="#f7c1c1", fontcolor="#501313", color="#a32d2d"];
  labw [label="Lab: wood risk\nconfirm before use", fillcolor="#fac775", fontcolor="#412402", color="#854f0b"];
  label [label="Label\npeanut and/or walnut", fillcolor="#b5d4f4", fontcolor="#042c53", color="#185fa5"];
  labp [label="Lab: verify pure", fillcolor="#faeeda", fontcolor="#412402", color="#854f0b"];
  record [label="6. Inspector decides\nand records it"];
  result [label="7. Lab result\non file"];
  {rank=same; lot; sample; ref;}
  {rank=same; models; rules;}
  {rank=same; reject; labw; label; labp;}
  {rank=same; record; result;}
  lot -> sample -> ref; ref -> models; models -> rules;
  rules -> reject; rules -> labw; rules -> label; rules -> labp;
  reject -> record; labw -> record; label -> record; labp -> record;
  record -> result;
  result -> ref [style=dashed, constraint=false, tooltip="Confirmed lots build the supplier's track record"];
}
"""


def before_after_chart():
    """Raw colour against colour relative to each supplier's reference."""
    s = S[~S.locked].copy()
    refs = {c: reference_lab(c) for c in C.FOLDERS}
    ref_rows = np.vstack([refs[c] for c in s.code])
    s["dL"], s["da"] = s.L - ref_rows[:, 0], s.a - ref_rows[:, 1]
    is_ref = IS_REF[~S.locked.to_numpy()]
    pure = (s.level == 0).to_numpy()
    left, right = st.columns(2, gap="large")
    for col, (x, y, title, xt, yt) in zip((left, right), (
            ("a", "L", "Raw colour", "a* (green → red)", "L* (dark → light)"),
            ("da", "dL", "Colour minus the supplier's reference", "a* difference", "L* difference"))):
        fig = go.Figure()
        for code in C.FOLDERS:
            m = (s.code == code).to_numpy() & ~pure
            fig.add_trace(go.Scatter(x=s[x][m], y=s[y][m], mode="markers", name=f"{TRUTH[code]} blends",
                                     marker=dict(size=5, color=FILLER_COLOUR[code], opacity=0.25),
                                     hovertemplate=f"{TRUTH[code]} blend<extra></extra>"))
        for code, sym in zip(C.FOLDERS, ("circle", "diamond", "square")):
            m = (s.code == code).to_numpy() & pure & (~is_ref if x == "da" else True)
            fig.add_trace(go.Scatter(x=s[x][m], y=s[y][m], mode="markers", name=f"Pure, {TRUTH[code].lower()} supplier",
                                     marker=dict(size=10, symbol=sym, color="#2c2c2a", line=dict(color="white", width=1.5)),
                                     hovertemplate=f"Pure cinnamon, {TRUTH[code].lower()} supplier<extra></extra>"))
        if x == "da":
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="The supplier's reference",
                                     marker=dict(size=16, symbol="star", color="#eda100", line=dict(color="white", width=1.5)),
                                     hovertemplate="Reference<extra></extra>"))
        with col:
            chart(fig, 380, title=title, xaxis=dict(title=xt, zeroline=False), yaxis=dict(title=yt, zeroline=False),
                  legend=dict(orientation="h", y=-0.25, font=dict(size=11)))
    st.caption("Each dot is one sample's colorimeter reading. **Left:** the three pure batches (dark shapes) sit far "
               "apart, so there is no single “pure cinnamon” colour to learn. **Right:** the same readings measured from "
               "each supplier's own reference (star). Every pure batch now lands on the star, so “how far from this "
               "supplier's normal” means the same thing for every supplier. Peanut-shell and wood-dust blends sit clearly "
               "away from the star in colour. Walnut-shell blends sit close to it in colour alone; the photo features are "
               "what separate those, which is why the screen uses colour and photo together.")


def page_how():
    st.title("How it works")
    st.info("**In one sentence:** the screen checks each delivery against a known-pure sample from the same supplier, "
            "estimates the chance of each filler, and turns that into one of four verdicts. It supplements lab "
            "testing; it does not replace it.")
    st.subheader("1 · The workflow")
    st.graphviz_chart(WORKFLOW_DOT, width="content")
    st.markdown(
        "1. **A lot arrives** from a supplier whose pure reference sample is already on file.\n"
        "2. **A small sample is ground and measured:** one photo in a dish and one colorimeter reading. No lab needed.\n"
        "3. **It is compared with that supplier's reference.** The screen looks at how far the sample is from the "
        "supplier's normal, not at its colour on its own.\n"
        "4. **Three simple models** estimate: is there any filler, which filler it would be, and roughly how much. "
        "They combine into a chance that each filler is present.\n"
        "5. **The verdict rules** set by the QC manager turn those chances into one verdict: reject, lab for wood risk, "
        "label, or lab to verify pure.\n"
        "6. **The inspector decides** and records the decision; the tool only recommends.\n"
        "7. **Lab results go on file.** Confirmed lots build each supplier's track record.")

    st.subheader("2 · Why every sample is compared with its supplier's reference")
    before_after_chart()

    st.subheader("3 · New suppliers and new kinds of cinnamon")
    ns = R["reference"]["new_supplier"]
    st.markdown(
        "Cinnamon differs by origin, species and batch, so the screen doesn't carry one fixed idea of “pure”. "
        "It carries the supplier's own reference. Bringing on a new supplier, or a new kind of cinnamon, works like this:\n\n"
        "1. **Get a reference sample** of the supplier's pure cinnamon and have the lab confirm it is pure.\n"
        "2. **Register it.** The screen measures it once and stores it as that supplier's normal.\n"
        "3. **Screen every lot against it.** The models don't need retraining: they learned how fillers move a "
        "sample away from its reference, which carries over to a new reference.\n"
        "4. **Check early lots at the lab too.** Each confirmed lot adds to the supplier's track record.\n"
        "5. **A different kind of cinnamon** (for example cassia instead of Ceylon) gets its own reference, "
        "even from the same supplier.")
    st.markdown("**How well this carries over, on this dataset.** Each supplier was held out completely: the models "
                "never saw its samples and were given only its reference.")
    rows = []
    for ss in ("colour+image", "all", "colour", "ftir"):
        v = ns[ss]
        rows.append({"Instruments": SET_LABEL[ss], "Blends flagged": pct(v["blends_flagged"], 0),
                     "Pure samples recognised": pct(v["pure_recognised"], 0),
                     "Blends flagged, by supplier": ", ".join(f"{TRUTH[c].split()[0]} {pct(x['blends_flagged'], 0)}"
                                                             for c, x in v["by_supplier"].items())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.warning(
        "**What this does not prove yet.**\n"
        "- The dataset has only three suppliers' pure batches, so “works for other cinnamon” is promising, not shown.\n"
        "- Some instruments carry over badly: FTIR alone missed the peanut blends of a new supplier.\n"
        "- The pure samples tested share a batch with their reference, so pure-cinnamon results are optimistic.\n"
        "- A filler the models have never seen (cassia, lead chromate, other dust) would still be named as one of the three.\n"
        "- A pilot would need many independent lots per supplier, each confirmed at the lab.")


# ---------------------------------------------------------------- Compare
def page_compare():
    st.title("Compare two samples")
    st.caption("Pick two samples to see their verdicts side by side, with the limits from the sidebar. Unlike Inspect, "
               "this page shows what each sample really is from the start.")
    df = VERD[VERD.pool == POOL].set_index("row")
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
    for side, col in zip("AB", st.columns(2, gap="large")):
        with col:
            st.subheader(f"Sample {side}")
            a, b = st.columns([2, 1])
            cell = a.selectbox("Material", cells, format_func=cell_name, key=f"cmp_cell_{side}")
            reps = sorted(S.rep.iloc[df.index[df.cell == cell]].astype(int))
            if st.session_state.get(f"cmp_rep_{side}") not in reps:
                st.session_state[f"cmp_rep_{side}"] = reps[0]
            rep_no = b.selectbox("Sample number", reps, key=f"cmp_rep_{side}",
                                 help="Half of each pure batch is kept as the supplier's reference and is not offered here.")
            cur = int(np.flatnonzero(((S.cell == cell) & (S.rep == rep_no)).to_numpy())[0])
            row, smp = df.loc[cur], S.iloc[cur]
            picked[side] = (cur, row, smp)
            st.image(str(C.THUMBS / f"{smp.sample_id}.jpg"), caption=describe(smp))
            lane_card(row)

    (ca, ra, sa), (cb, rb, sb) = picked["A"], picked["B"]
    st.divider()
    if ca == cb:
        st.info("Both sides show the same sample.")
    elif ra.lane == rb.lane and label_text(ra) == label_text(rb):
        st.success(f"Both samples get the same verdict: **{LANE_CHOICE[ra.lane]}**.")
    else:
        st.warning(f"The verdicts differ: sample A gets **{LANE_CHOICE[ra.lane]}**, sample B gets **{LANE_CHOICE[rb.lane]}**.")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.dataframe(styled({"Sample A": sample_rows(ra, sa, ca, True, extras=True),
                             "Sample B": sample_rows(rb, sb, cb, True, extras=True)}), width="stretch")
    with right:
        fig = go.Figure()
        names = ["Peanut shell", "Walnut shell", "Wood dust"]
        for name, row, colr in (("Sample A", ra, "#2a78d6"), ("Sample B", rb, "#eb6834")):
            vals = [row.p_peanut, row.p_walnut, row.p_wood]
            fig.add_trace(go.Bar(y=names, x=vals, orientation="h", name=name, marker_color=colr,
                                 text=[pct(v, 0) for v in vals], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x:.1%}<extra>" + name + "</extra>"))
        chart(fig, 300, title="Chance each filler is present", barmode="group", bargap=0.3,
              xaxis=dict(range=[0, 1.12], tickformat=".0%", showgrid=False), yaxis=dict(autorange="reversed"),
              legend=dict(orientation="h", y=-0.2, traceorder="normal"))


# ---------------------------------------------------------------- Scoreboard
OUTCOME_ORDER = ["✕ Reject", "⚠ Lab (wood risk)", "🏷 Label peanut", "🏷 Label walnut", "🏷 Label both", "⧖ Lab (verify pure)"]


def outcome_name(row) -> str:
    if row.lane == V.LABEL:
        return "🏷 Label both" if row.label_peanut and row.label_walnut else (
            "🏷 Label peanut" if row.label_peanut else "🏷 Label walnut")
    return LANE_CHOICE[row.lane]


def verdict_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    truth = np.where(df.level == 0, "Pure cinnamon", df.code.map(TRUTH))
    t = pd.crosstab(pd.Series(truth, name="What it really is"), pd.Series(df.apply(outcome_name, axis=1), name="Verdict"))
    return t.reindex(index=["Wood dust", "Peanut shell", "Walnut shell", "Pure cinnamon"],
                     columns=[c for c in OUTCOME_ORDER if c in t.columns], fill_value=0)


OUTCOME_COLOUR = {"✕ Reject": "#d03b3b", "⚠ Lab (wood risk)": "#ec835a", "🏷 Label peanut": "#2a78d6",
                  "🏷 Label walnut": "#6250d6", "🏷 Label both": "#1baf7a", "⧖ Lab (verify pure)": "#fab219"}


def verdict_bars(ct: pd.DataFrame):
    """One bar per kind of sample, split by the verdict it got."""
    share = ct.div(ct.sum(axis=1), axis=0)
    fig = go.Figure()
    for col in ct.columns:
        fig.add_trace(go.Bar(y=ct.index, x=share[col], name=col, orientation="h", marker_color=OUTCOME_COLOUR[col],
                             marker_line=dict(color="rgba(0,0,0,0)", width=0), customdata=ct[col],
                             text=[pct(v, 0) if v >= 0.06 else "" for v in share[col]], textposition="inside",
                             hovertemplate="%{y} → " + col + ": %{customdata} samples (%{x:.0%})<extra></extra>"))
    chart(fig, 260, barmode="stack", bargap=0.35, title="What each kind of sample was told",
          xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False), yaxis=dict(autorange="reversed"),
          legend=dict(orientation="h", y=-0.2))


def page_scoreboard():
    st.title("Scoreboard")
    st.caption(f"How the verdicts do on materials the models had never seen. Instruments: {SET_LABEL[SENSORS]} · limits: "
               f"wood lab {WOOD_LAB}%, wood reject {WOOD_REJECT}%, peanut {PEANUT}%, walnut {WALNUT}%.")
    st.info("**In plain words:** each sample here was judged by models that had never seen that material, and compared "
            "with its supplier's reference. Pure-cinnamon results are optimistic: the pure samples tested come from the "
            "same batch as the reference.")
    dev = VERD[VERD.pool == "dev"]
    m = V.summary(dev)
    good = m["wood_let_through"] == 0 and m["filler_missed"] == 0
    st.markdown(f"### {'✓' if good else '⚠'} On materials it had never seen, the screen "
                + (f"let no wood through and missed no filler. {pct(m['right'])} of verdicts were right." if good else
                   f"let {pct(m['wood_let_through'])} of wood through and missed {pct(m['filler_missed'])} of filler. "
                   f"{pct(m['right'])} of verdicts were right."))
    st.subheader("1 · At a glance")
    cards = [
        ("Wood let through", m["wood_let_through"], m["wood_let_through"] == 0, "target 0%",
         "Wood dust blends that were neither rejected nor sent to the lab."),
        ("Filler missed", m["filler_missed"], m["filler_missed"] == 0, "target 0%",
         "Adulterated samples sent to the lab as if they might be pure."),
        ("Peanut labelled", m["peanut_labelled"], m["peanut_labelled"] >= 0.95, "target ≥ 95%",
         "Peanut-shell blends labelled peanut (and not flagged as wood)."),
        ("Walnut labelled", m["walnut_labelled"], m["walnut_labelled"] >= 0.95, "target ≥ 95%",
         "Walnut-shell blends labelled walnut (and not flagged as wood)."),
        ("Pure sent to verify", m["pure_to_verify"], m["pure_to_verify"] >= 0.95, "target ≥ 95%",
         "Pure cinnamon that went to the lab to verify, rather than being flagged."),
        ("Verdicts right", m["right"], m["right"] >= 0.95, "target ≥ 95%", "All samples whose verdict matched what they are."),
    ]
    for col, (name, v, ok, target, text) in zip(st.columns(len(cards)), cards):
        col.metric(name, pct(v))
        col.caption(f"{'✓' if ok else '✗'} {target}. {text}")

    st.subheader("2 · Every verdict, by what the sample really is")
    verdict_bars(verdict_crosstab(dev))
    st.dataframe(verdict_crosstab(dev), width="stretch")
    st.caption(f"{m['n']:,} samples from {m['materials']} materials. Rows are the truth, columns the verdict.")

    st.subheader("3 · Easy test against fair test")
    st.caption("Easy: the model has seen near-copies of the test samples (random split). Fair: it has never seen the "
               "material. The drop shows how much of the easy score came from recognising repeats. These are the "
               "underlying models, compared with pure cinnamon in general.")
    ev = easy_vs_fair(pd.DataFrame(R["rows"]))
    st.dataframe(ev, hide_index=True, width="stretch")

    st.subheader("4 · Final test: the locked materials")
    if not SCORED:
        st.warning("🔒 **Not scored yet.** Levels 3%, 10% and 18% of every filler (9 materials, 450 samples) have never been "
                   "used for training, tuning or any number on this page. They are scored once, at the end. Those are the "
                   "numbers to quote.")
        st.code(".venv/bin/python -m earnedpass.run_all --score-locked", language="bash")
    else:
        lk = VERD[VERD.pool == "locked"]
        st.success(f"Scored once at {R['meta']['locked_scored_at']}. These are the numbers to quote.")
        st.dataframe(verdict_crosstab(lk), width="stretch")
        lm = V.summary(lk)
        st.caption(f"Wood let through {pct(lm['wood_let_through'])} · filler missed {pct(lm['filler_missed'])} · "
                   f"verdicts right {pct(lm['right'])}.")
    st.download_button("Download the verdict table (CSV)", verdict_crosstab(dev).to_csv().encode(),
                       "earned_pass_verdicts.csv", "text/csv")


def easy_vs_fair(rows: pd.DataFrame) -> pd.DataFrame:
    lad = rows[rows.section == "ladder"]
    spec = [("detect", "balanced_accuracy", "Pure vs adulterated (balanced accuracy)", 2),
            ("type", "macro_f1", "Which filler (macro-F1)", 2),
            ("level", "mae_pct_points", "Amount error (points)", 1)]
    out = []
    for ss in C.SENSOR_SETS:
        rec = {"Instruments": SET_LABEL[ss]}
        for task, metric, name, dg in spec:
            def get(rung):
                v = lad[(lad.task == task) & (lad.sensor_set == ss) & (lad.rung == rung) & (lad.metric == metric)].value
                return float(v.iloc[0]) if len(v) else float("nan")
            rec[name] = f"{get('1_sample_id'):.{dg}f} → {get('3_leave_cell_out'):.{dg}f}"
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Evidence
def tab_why_reference():
    st.info("**In plain words:** the same verdict rules, run two ways. Compared with pure cinnamon in general, the "
            "screen mistakes unfamiliar pure cinnamon for filler. Compared with each supplier's own reference sample, "
            "it doesn't. That is why the screen always uses the supplier's reference.")
    gen = verdicts(STAMP, "general", SENSORS, TH)
    gen = gen[gen.pool == "dev"]
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Compared with pure cinnamon in general**")
        st.dataframe(verdict_crosstab(gen), width="stretch")
        st.caption(f"Verdicts right: {pct(V.summary(gen)['right'])}.")
    with b:
        st.markdown("**Compared with the supplier's reference**")
        sup = VERD[VERD.pool == "dev"]
        st.dataframe(verdict_crosstab(sup), width="stretch")
        st.caption(f"Verdicts right: {pct(V.summary(sup)['right'])}.")
    ns = R["reference"]["new_supplier"][SENSORS]
    st.markdown(
        "**Why general mode fails.** Each pure batch looks different, and pure Peanut-folder cinnamon is much lighter "
        "than any blend. A model that has never seen that batch extrapolates and gives it a near-100% chance of wood.\n\n"
        f"**Limit of supplier mode.** It needs a reference for every supplier. For a supplier the model never saw "
        f"(given only its reference), {pct(ns['blends_flagged'], 0)} of blends were still flagged and "
        f"{pct(ns['pure_recognised'], 0)} of pure samples recognised.")


def tab_blind_spot():
    st.info("**In plain words:** each square is one material the screen had never seen. Green: at least 95% of its "
            "samples got the right verdict. Red: fewer than half did. Grey in between. White squares are set aside for "
            "the final test.")
    d = VERD[VERD.pool == "dev"]
    share = d.assign(ok=d.result == "right").groupby(["code", "level"]).ok.mean()
    z, text = [], []
    for code in C.FOLDERS:
        zr, tr = [], []
        for level in C.LEVELS:
            if (code, level) in share.index:
                v = share[(code, level)]
                zr.append(0 if v >= 0.95 else (2 if v < 0.5 else 1))
                tr.append(f"{'✓' if v >= 0.95 else ('✗' if v < 0.5 else '◐')} {v:.0%}")
            else:
                zr.append(3)
                tr.append("locked")
        z.append(zr)
        text.append(tr)
    scale = [[0, "#0ca30c"], [0.25, "#0ca30c"], [0.25, "#c3c2b7"], [0.5, "#c3c2b7"], [0.5, "#d03b3b"], [0.75, "#d03b3b"],
             [0.75, "#f0efec"], [1, "#f0efec"]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{l}%" for l in C.LEVELS], y=[C.FOLDER_LABEL[c] for c in C.FOLDERS], text=text,
                               texttemplate="%{text}", colorscale=scale, zmin=0, zmax=4, showscale=False, xgap=3, ygap=3,
                               hovertemplate="%{y} at %{x}: %{text}<extra></extra>"))
    chart(fig, 260, title=f"Share of each material's samples with the right verdict ({SET_LABEL[SENSORS]})",
          xaxis=dict(type="category", title="Adulteration level (% w/w)"), yaxis=dict(autorange="reversed"))


def page_evidence():
    st.title("Evidence")
    st.caption(f"Instruments: {SET_LABEL[SENSORS]} · limits from the sidebar.")
    tabs = st.tabs(["Why supplier reference", "Blind-spot map", "Replicate Radar", "Stress ladder", "Controls"])
    for tab, fn in zip(tabs, (tab_why_reference, tab_blind_spot, tab_radar, tab_ladder, tab_controls)):
        with tab:
            fn()


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
    st.Page(page_how, title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
    st.Page(page_compare, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(page_scoreboard, title="Scoreboard", icon=":material/scoreboard:", url_path="scoreboard"),
    st.Page(page_evidence, title="Evidence", icon=":material/fact_check:", url_path="evidence"),
    st.Page(page_log, title="Log", icon=":material/list_alt:", url_path="log"),
]).run()
