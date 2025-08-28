# app.py — NYC 311 Insights (senior-level)
# - Token-free mapping (Plotly open-street-map)
# - Fast, cache-first data loading
# - KPIs + robust Actual vs Forecast overlay
# - Anomaly markers (if anomalies.csv present)
# - Borough mix analysis, DOW heatmap, and flexible filters

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NYC 311 Insights", layout="wide")

# -----------------------------
# Helpers & cached data loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_daily():
    """Load daily aggregates from data/train.csv & data/test.csv."""
    dfs = []
    for p in ["data/train.csv", "data/test.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            dfs.append(df)
    if len(dfs) == 0:
        # Fallback demo data (reproducible without internet)
        dts = pd.date_range("2025-05-01", "2025-08-01", freq="D")
        df = pd.DataFrame({"date": dts, "total_calls": 10000})
        for b in ["manhattan","bronx","brooklyn","queens","staten_island"]:
            df[b] = (df["total_calls"]/5).astype(int)
        return df
    df = pd.concat(dfs, ignore_index=True).sort_values("date")
    # Fill borough columns if missing
    for b in ["manhattan","bronx","brooklyn","queens","staten_island"]:
        if b not in df.columns:
            df[b] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_submission():
    """Load submission (forecast) if present."""
    if os.path.exists("submission.csv"):
        sub = pd.read_csv("submission.csv")
        sub["Date"] = pd.to_datetime(sub["Date"])
        sub.rename(columns={"Date":"date","Predicted_Total_Calls":"forecast"}, inplace=True)
        return sub
    return pd.DataFrame(columns=["date","forecast"])

@st.cache_data(show_spinner=False)
def load_anomalies():
    """Load anomalies if present."""
    if os.path.exists("anomalies.csv"):
        an = pd.read_csv("anomalies.csv")
        if "Date" in an.columns: an["Date"] = pd.to_datetime(an["Date"])
        return an
    return pd.DataFrame(columns=["Date","Actual","Expected","Anomaly_Score","Note"])

def kpi_delta(cur, prev):
    if prev in (0, None) or np.isnan(prev): 
        return 0.0
    return 100.0 * (cur - prev) / max(prev, 1e-6)

# -------------
# Load datasets
# -------------
daily = load_daily()
sub   = load_submission()
anom  = load_anomalies()

# Harmonize
df = daily.copy()
df["date"] = pd.to_datetime(df["date"])
if "total_calls" not in df.columns:
    st.stop()

# Join forecast if available
df = df.merge(sub, how="left", on="date")

# -----------
# UI Controls
# -----------
st.title("NYC 311 Insights Dashboard")

with st.sidebar:
    st.header("Filters")
    min_d, max_d = df["date"].min(), df["date"].max()
    dr = st.date_input("Date range", [min_d, max_d], min_value=min_d, max_value=max_d)
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

    # Borough selector (only those present)
    borough_cols = [c for c in ["manhattan","bronx","brooklyn","queens","staten_island"] if c in df.columns]
    borough_sel = st.multiselect("Boroughs", borough_cols, default=borough_cols)

    # Moving average toggle
    ma_window = st.slider("Moving average (days)", 1, 21, 7)

mask = df["date"].between(start, end)
view = df.loc[mask].copy()

# KPIs
view["ma_calls"] = view["total_calls"].rolling(ma_window).mean()
week_ago = view["date"].max() - pd.Timedelta(days=7)
cur = view.loc[view["date"]==view["date"].max(),"total_calls"].squeeze()
prev = view.loc[view["date"]==week_ago,"total_calls"].squeeze() if (view["date"]==week_ago).any() else np.nan
delta = kpi_delta(cur, prev)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest day calls", f"{int(cur) if pd.notna(cur) else '-'}", f"{delta:+.1f}% vs 7d ago")
col2.metric("Range avg (MA)", f"{int(view['ma_calls'].mean()):,}" if not view['ma_calls'].isna().all() else "-")
col3.metric("Range max", f"{int(view['total_calls'].max()):,}")
col4.metric("Range min", f"{int(view['total_calls'].min()):,}")

# -----------------------
# Visual 1: Time-series
# -----------------------
ts = go.Figure()
ts.add_trace(go.Scatter(
    x=view["date"], y=view["total_calls"], name="Actual", mode="lines",
    hovertemplate="Date=%{x|%Y-%m-%d}<br>Calls=%{y:,.0f}<extra></extra>"
))
if "forecast" in view.columns and view["forecast"].notna().any():
    ts.add_trace(go.Scatter(
        x=view["date"], y=view["forecast"], name="Forecast", mode="lines",
        line=dict(dash="dash"),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Forecast=%{y:,.0f}<extra></extra>"
    ))
# Optional MA
if ma_window > 1:
    ts.add_trace(go.Scatter(
        x=view["date"], y=view["total_calls"].rolling(ma_window).mean(),
        name=f"{ma_window}-day MA", mode="lines",
        hovertemplate="Date=%{x|%Y-%m-%d}<br>MA=%{y:,.0f}<extra></extra>"
    ))
# Anomaly markers if available
if not anom.empty:
    a = anom.copy()
    a["Date"] = pd.to_datetime(a["Date"])
    a = a[(a["Date"]>=start) & (a["Date"]<=end)]
    if not a.empty and "Anomaly_Score" in a.columns:
        ts.add_trace(go.Scatter(
            x=a["Date"], y=a["Actual"], mode="markers",
            marker=dict(size=10, color=np.where(a["Anomaly_Score"]>=0, "crimson", "teal")),
            name="Anomalies",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Actual=%{y:,.0f}<br>Δ%="+
                          a["Anomaly_Score"].map(lambda x: f"{x*100:.1f}%").astype(str)+
                          "<br>"+a.get("Note","").astype(str)+"<extra></extra>"
        ))

ts.update_layout(title="Daily 311 Calls — Actual vs Forecast (with optional MA & anomalies)",
                 xaxis_title="", yaxis_title="Calls", legend_title="")
st.plotly_chart(ts, use_container_width=True)

# --------------------------------------
# Visual 2: Borough mix / contribution
# --------------------------------------
if borough_sel:
    mix = view[["date"]+borough_sel].copy()
    # If borough columns are NaN (because source lacked them), derive simple split
    if mix[borough_sel].isna().all().all():
        share = 1.0/len(borough_sel)
        for b in borough_sel:
            mix[b] = (view["total_calls"]*share).round()
    fig_mix = px.area(mix, x="date", y=borough_sel, title="Borough Breakdown (stacked area)")
    st.plotly_chart(fig_mix, use_container_width=True)

# --------------------------------------
# Visual 3: Day-of-Week heatmap
# --------------------------------------
dow = view.copy()
dow["dow"] = dow["date"].dt.day_name()
dow["week"] = dow["date"].dt.isocalendar().week
pivot = dow.pivot_table(index="dow", columns="week", values="total_calls", aggfunc="mean")
# Order weekdays Mon..Sun
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
pivot = pivot.reindex(weekday_order)
fig_heat = px.imshow(pivot, aspect="auto", title="Intensity by Day-of-Week vs Week",
                     labels=dict(x="ISO Week", y="Day of Week", color="Calls"))
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------
# Visual 4: Token-free Map
# ---------------------------
# We don't have point-level lat/lon in daily aggregates; build a borough centroid demo
borough_points = pd.DataFrame({
    "borough": ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"],
    "lat": [40.7831, 40.6782, 40.7282, 40.8448, 40.5795],
    "lon": [-73.9712,-73.9442,-73.7949,-73.8648,-74.1502]
})
# attach counts from the selected range
if borough_sel:
    bsum = view[borough_sel].sum().rename_axis("borough").reset_index(name="count")
else:
    # use proportional split from total_calls if columns missing
    share = 1.0/5
    bsum = pd.DataFrame({"borough": borough_points["borough"],
                         "count": [int(view["total_calls"].sum()*share)]*5})
map_df = borough_points.merge(bsum, on="borough", how="left").fillna(0)

fig_map = px.scatter_mapbox(
    map_df, lat="lat", lon="lon", size="count", color="borough",
    hover_data={"count":":,", "lat":False, "lon":False},
    zoom=9, height=500, title="NYC Borough Activity (token-free base map)",
    mapbox_style="open-street-map"
)
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# Insights callout
# ---------------------------
with st.expander("Quick insights"):
    bullet = []
    if "forecast" in view.columns and view["forecast"].notna().any():
        err = (view["total_calls"]-view["forecast"]).abs().mean()
        bullet.append(f"Avg absolute error in selected range: **{err:,.0f}** calls.")
    wd = view.groupby(view["date"].dt.day_name())["total_calls"].mean().sort_values(ascending=False)
    bullet.append(f"Highest average by weekday: **{wd.index[0]}** ({wd.iloc[0]:,.0f}).")
    st.write("• " + "\n\n• ".join(bullet))
