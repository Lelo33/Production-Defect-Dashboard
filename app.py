"""Production Defect Analysis Dashboard - Pareto, p-chart and line/shift heatmap."""
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REQUIRED = ["Date", "Line", "Shift", "Product", "Defect type", "Inspected", "Defects"]
LINES = ["Line 1", "Line 2", "Line 3"]
SHIFTS = ["Day", "Afternoon", "Night"]
PRODUCTS = ["Product A", "Product B", "Product C"]
DEFECT_TYPES = ["Dimension out of tolerance", "Surface scratch", "Label misprint",
                "Contamination", "Assembly error", "Packaging damage"]

TEMPLATE_CSV = (
    "Date,Line,Shift,Product,Defect type,Inspected,Defects\n"
    "2026-01-05,Line 1,Day,Product A,Surface scratch,500,6\n"
    "2026-01-05,Line 1,Day,Product A,Label misprint,0,3\n"
    "2026-01-05,Line 2,Night,Product B,None,480,0\n"
    "2026-01-06,Line 2,Night,Product B,Contamination,450,9\n"
)


@st.cache_data
def sample_data(days=60, seed=42):
    """Simulated production data. One batch per day, line and shift."""
    rng = np.random.default_rng(seed)
    today = date.today()
    base = {"Line 1": 0.008, "Line 2": 0.010, "Line 3": 0.015}
    shift_factor = {"Day": 0.8, "Afternoon": 1.0, "Night": 1.4}
    weights = {
        "Line 1": [0.35, 0.20, 0.10, 0.10, 0.15, 0.10],
        "Line 2": [0.15, 0.15, 0.35, 0.10, 0.15, 0.10],
        "Line 3": [0.40, 0.25, 0.05, 0.10, 0.10, 0.10],
    }
    rows = []
    for d in range(days, 0, -1):
        day = today - timedelta(days=d)
        for line in LINES:
            for shift in SHIFTS:
                n = int(rng.integers(400, 601))
                p = base[line] * shift_factor[shift]
                if line == "Line 2" and 20 <= d <= 24:
                    p *= 4.0  # a deliberate special-cause spike
                defects = int(rng.binomial(n, p))
                product = str(rng.choice(PRODUCTS))
                if defects == 0:
                    rows.append([day, line, shift, product, "None", n, 0])
                    continue
                split = rng.multinomial(defects, weights[line])
                first = True
                for name, count in zip(DEFECT_TYPES, split):
                    if count > 0:
                        rows.append([day, line, shift, product, name,
                                     n if first else 0, int(count)])
                        first = False
    out = pd.DataFrame(rows, columns=REQUIRED)
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def load_upload(file):
    """Read and validate an uploaded CSV. Returns (dataframe, error message)."""
    try:
        raw = pd.read_csv(file)
    except Exception:
        return None, "Could not read that file as a CSV."
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)
    if len(raw) > 100000:
        return None, "Please upload 100,000 rows or fewer."
    df = raw[REQUIRED].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Inspected", "Defects"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Line", "Shift", "Product", "Defect type"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["Date", "Inspected", "Defects"])
    if df.empty:
        return None, "No valid rows found. Check the dates and numbers."
    if (df["Inspected"] < 0).any() or (df["Defects"] < 0).any():
        return None, "Inspected and Defects cannot be negative."
    if df["Defects"].sum() > df["Inspected"].sum():
        return None, "Total defects are larger than total inspected. Check your columns."
    return df, None


def defect_rate(frame):
    inspected = frame["Inspected"].sum()
    return frame["Defects"].sum() / inspected if inspected > 0 else float("nan")


def pareto_chart(df):
    counts = (df[df["Defect type"] != "None"].groupby("Defect type")["Defects"]
              .sum().sort_values(ascending=False))
    counts = counts[counts > 0]
    if counts.empty:
        return None
    cumulative = counts.cumsum() / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts.values, color="tab:blue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=35, ha="right")
    ax.set_ylabel("Defects")
    ax2 = ax.twinx()
    ax2.plot(range(len(counts)), cumulative.values, color="tab:red", marker="o")
    ax2.axhline(80, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative %")
    fig.tight_layout()
    return fig


def p_chart_data(df):
    """Daily defect rate with 3-sigma p-chart limits (limits vary with daily volume)."""
    daily = df.groupby("Date", as_index=False)[["Inspected", "Defects"]].sum()
    daily = daily[daily["Inspected"] > 0].copy()
    if daily.empty:
        return None, None
    daily["Rate"] = daily["Defects"] / daily["Inspected"]
    p_bar = daily["Defects"].sum() / daily["Inspected"].sum()
    se = np.sqrt(p_bar * (1 - p_bar) / daily["Inspected"])
    daily["UCL"] = p_bar + 3 * se
    daily["LCL"] = (p_bar - 3 * se).clip(lower=0)
    daily["Out of control"] = (daily["Rate"] > daily["UCL"]) | (daily["Rate"] < daily["LCL"])
    return daily, p_bar


def p_chart(daily, p_bar):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(daily["Date"], daily["Rate"] * 100, marker="o", markersize=3,
            color="tab:blue", label="Daily defect rate")
    ax.axhline(p_bar * 100, color="green", label="Average")
    ax.step(daily["Date"], daily["UCL"] * 100, where="mid", color="red",
            linestyle="--", label="UCL / LCL")
    ax.step(daily["Date"], daily["LCL"] * 100, where="mid", color="red", linestyle="--")
    flagged = daily[daily["Out of control"]]
    if not flagged.empty:
        ax.scatter(flagged["Date"], flagged["Rate"] * 100, color="red", s=60,
                   zorder=5, label="Out of control")
    ax.set_ylabel("Defect rate (%)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def heatmap(df):
    agg = df.groupby(["Line", "Shift"])[["Inspected", "Defects"]].sum()
    agg["Rate"] = np.where(agg["Inspected"] > 0, agg["Defects"] / agg["Inspected"] * 100, np.nan)
    table = agg["Rate"].unstack("Shift")
    cols = [s for s in SHIFTS if s in table.columns] + [c for c in table.columns if c not in SHIFTS]
    table = table[cols]
    fig, ax = plt.subplots(figsize=(6, 1.2 + 0.7 * len(table)))
    im = ax.imshow(table.values.astype(float), cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    top = np.nanmax(table.values.astype(float)) if np.isfinite(table.values.astype(float)).any() else 1
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            label = "-" if pd.isna(v) else f"{v:.2f}%"
            dark = (not pd.isna(v)) and v > top * 0.6
            ax.text(j, i, label, ha="center", va="center", color="white" if dark else "black")
    fig.colorbar(im, ax=ax, label="Defect rate (%)")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Production Defect Dashboard")
st.title("Production Defect Analysis Dashboard")
st.caption(
    "Pareto, p-chart and line/shift heatmap for production defect data. "
    "The sample data is simulated. This app is public, so do not upload confidential company data."
)

source = st.radio("Data source", ["Sample data", "Upload a CSV"], horizontal=True)
if source == "Sample data":
    data = sample_data()
else:
    st.download_button("Download CSV template", TEMPLATE_CSV,
                       file_name="defect_template.csv", mime="text/csv")
    st.caption(
        "Each row is an inspection record. Inspected is the quantity checked and Defects is the "
        "number found of that Defect type. If a batch has several defect types, put the Inspected "
        "quantity on the first row and 0 on the others. Use 'None' when no defects were found."
    )
    uploaded = st.file_uploader("Upload your CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with the columns: " + ", ".join(REQUIRED))
        st.stop()
    data, error = load_upload(uploaded)
    if error:
        st.error(error)
        st.stop()

with st.expander("Filters"):
    lines = st.multiselect("Line", sorted(data["Line"].unique()), default=sorted(data["Line"].unique()))
    shifts = st.multiselect("Shift", sorted(data["Shift"].unique()), default=sorted(data["Shift"].unique()))
    products = st.multiselect("Product", sorted(data["Product"].unique()), default=sorted(data["Product"].unique()))
    first_day, last_day = data["Date"].min().date(), data["Date"].max().date()
    window = st.date_input("Date range", value=(first_day, last_day),
                           min_value=first_day, max_value=last_day)

if not isinstance(window, (tuple, list)) or len(window) != 2:
    st.info("Choose both a start date and an end date.")
    st.stop()

mask = (data["Line"].isin(lines) & data["Shift"].isin(shifts) & data["Product"].isin(products)
        & (data["Date"].dt.date >= window[0]) & (data["Date"].dt.date <= window[1]))
df = data[mask]
if df.empty or df["Inspected"].sum() == 0:
    st.warning("No inspection records match these filters.")
    st.stop()

rate = defect_rate(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Inspected", f"{int(df['Inspected'].sum()):,}")
c2.metric("Defects", f"{int(df['Defects'].sum()):,}")
c3.metric("Defect rate", f"{rate * 100:.2f}%")
c4.metric("PPM", f"{rate * 1_000_000:,.0f}")

by_line = {k: defect_rate(g) for k, g in df.groupby("Line")}
by_shift = {k: defect_rate(g) for k, g in df.groupby("Shift")}
worst_line = max(by_line, key=lambda k: by_line[k] if not np.isnan(by_line[k]) else -1)
worst_shift = max(by_shift, key=lambda k: by_shift[k] if not np.isnan(by_shift[k]) else -1)
st.write(f"**Highest defect rate:** {worst_line} ({by_line[worst_line] * 100:.2f}%) "
         f"and {worst_shift} shift ({by_shift[worst_shift] * 100:.2f}%)")

st.subheader("Defect Pareto")
fig = pareto_chart(df)
if fig is None:
    st.info("No defects in this selection.")
else:
    st.pyplot(fig)
    st.caption("Bars show defects by type. The red line is the cumulative percentage and "
               "the dashed line marks 80%: the types left of it are the 'vital few'.")

st.subheader("Daily defect rate (p-chart)")
daily, p_bar = p_chart_data(df)
st.pyplot(p_chart(daily, p_bar))
st.caption("The green line is the average defect rate. Red dashed lines are 3-sigma control "
           "limits, which move with each day's volume. A red dot is a day that is out of control "
           "and worth investigating.")
flagged = daily[daily["Out of control"]]
if flagged.empty:
    st.success("No out-of-control days in this selection.")
else:
    show = flagged[["Date", "Inspected", "Defects", "Rate"]].copy()
    show["Date"] = show["Date"].dt.date
    show["Rate"] = (show["Rate"] * 100).round(2).astype(str) + "%"
    st.warning(f"{len(flagged)} out-of-control day(s):")
    st.dataframe(show, hide_index=True)

st.subheader("Defect rate by line and shift")
st.pyplot(heatmap(df))
