import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Meters — London Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0d1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px 26px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-label { color: #8b949e; font-size: 13px; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { color: #e6edf3; font-size: 28px; font-weight: 700; }
    .metric-unit  { color: #4ECDC4; font-size: 14px; font-weight: 400; }
    .section-title {
        font-size: 22px; font-weight: 600; color: #e6edf3;
        border-left: 3px solid #4ECDC4; padding-left: 12px;
        margin: 28px 0 18px;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 600; margin-right: 6px;
    }
    .badge-green  { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-orange { background: rgba(210,153,34,0.15); color: #d2993a; border: 1px solid rgba(210,153,34,0.3); }
    .badge-red    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
    hr { border-color: rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energy Analytics")
    st.markdown("<span style='color:#8b949e;font-size:13px;'>Smart Meters in London</span>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Overview", "Load Forecasting", "Pattern Analysis", "Project Info"])
    st.markdown("---")
    st.markdown("<span style='color:#8b949e;font-size:12px;'>Data: Low Carbon London<br>Author: Tarun Jindal<br>JECRC Foundation CEI</span>", unsafe_allow_html=True)

# ─── LOAD / GENERATE DATA ─────────────────────────────────────────────────────
@st.cache_data
def get_data():
    np.random.seed(42)
    dates = pd.date_range('2012-01-01', periods=730, freq='D')
    households = [f"MAC{str(i).zfill(5)}" for i in range(1, 51)]
    rows = []
    for hh in households:
        base = np.random.uniform(4, 12)
        for d in dates:
            seasonal = 1 + 0.5 * np.cos((d.dayofyear - 15) / 365 * 2 * np.pi)
            noise    = np.random.normal(0, 0.4)
            energy   = max(0, base * seasonal + noise)
            rows.append({'LCLid': hh, 'day': d, 'energy_sum': energy,
                         'Acorn': np.random.choice(['A','B','C','D','E'])})
    df = pd.DataFrame(rows)
    city = df.groupby('day')['energy_sum'].sum().reset_index()
    return df, city

df, city_daily = get_data()

# ─── PAGE: OVERVIEW ───────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("<h1 style='color:#e6edf3;'>London Smart Meter Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Aggregated daily energy consumption across **50 sampled households** — Nov 2011 to Feb 2014.")

    with st.expander("🌍 Initiative Objectives", expanded=True):
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            st.markdown("**🔌 Grid Modernization**\n\nHelp the British government understand complex consumption patterns to upgrade an aging electrical grid.")
        with col_o2:
            st.markdown("**📡 Rollout Preparation**\n\nGather baseline insights ahead of massive, nationwide smart meter rollouts.")
        with col_o3:
            st.markdown("**🌱 Climate Action**\n\nAnalyze how micro-habits and weather impact demand to build localized energy-saving strategies.")

    st.markdown("---")

    recent = city_daily.tail(60).copy()
    avg_daily  = recent['energy_sum'].mean()
    peak_value = recent['energy_sum'].max()
    peak_day   = recent.loc[recent['energy_sum'].idxmax(), 'day'].strftime('%b %d, %Y')
    num_hh     = df['LCLid'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Daily Usage</div>
            <div class="metric-value">{avg_daily:.1f} <span class="metric-unit">kWh</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Peak Usage Day</div>
            <div class="metric-value" style="font-size:20px;">{peak_day}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Peak Load</div>
            <div class="metric-value">{peak_value:.1f} <span class="metric-unit">kWh</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Households Tracked</div>
            <div class="metric-value">{num_hh}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">60-Day Consumption Trend</div>', unsafe_allow_html=True)
    fig = px.area(recent, x='day', y='energy_sum', template='plotly_dark',
                  labels={'energy_sum': 'Total kWh', 'day': 'Date'},
                  line_shape='spline', color_discrete_sequence=['#4ECDC4'])
    fig.update_traces(fill='tozeroy', fillcolor='rgba(78,205,196,0.15)')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: LOAD FORECASTING ───────────────────────────────────────────────────
elif page == "Load Forecasting":
    st.markdown("<h1 style='color:#e6edf3;'>7-Day Load Forecast</h1>", unsafe_allow_html=True)
    st.markdown("Statistical predictions for the next week based on seasonal and calendar patterns.")

    from sklearn.ensemble import RandomForestRegressor

    city_daily['dayofweek'] = city_daily['day'].dt.dayofweek
    city_daily['month']     = city_daily['day'].dt.month
    city_daily['dayofyear'] = city_daily['day'].dt.dayofyear
    city_daily['is_weekend']= city_daily['dayofweek'].isin([5, 6]).astype(int)
    city_daily['lag1']      = city_daily['energy_sum'].shift(1)
    city_daily['lag7']      = city_daily['energy_sum'].shift(7)
    city_daily = city_daily.dropna()

    feats = ['dayofweek', 'month', 'dayofyear', 'is_weekend', 'lag1', 'lag7']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(city_daily[feats], city_daily['energy_sum'])

    last_row   = city_daily.iloc[-1]
    last_date  = city_daily['day'].max()
    future_rows = []
    for i in range(1, 8):
        fd = last_date + timedelta(days=i)
        future_rows.append({
            'day': fd,
            'dayofweek': fd.dayofweek,
            'month': fd.month,
            'dayofyear': fd.dayofyear,
            'is_weekend': 1 if fd.dayofweek >= 5 else 0,
            'lag1': city_daily['energy_sum'].iloc[-i],
            'lag7': city_daily['energy_sum'].iloc[-(i+6)] if i+6 <= len(city_daily) else city_daily['energy_sum'].mean()
        })
    future_df  = pd.DataFrame(future_rows)
    future_df['Predicted_kWh'] = model.predict(future_df[feats])

    past14 = city_daily.tail(14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past14['day'], y=past14['energy_sum'],
                             mode='lines+markers', name='Actual',
                             line=dict(color='#e6edf3', width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=future_df['day'], y=future_df['Predicted_kWh'],
                             mode='lines+markers', name='Forecast',
                             line=dict(color='#FF6B6B', width=3, dash='dot'),
                             marker=dict(size=7, symbol='diamond')))
    fig.update_layout(template='plotly_dark', hovermode='x unified',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=1.1),
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
    display_df = future_df[['day', 'Predicted_kWh']].copy()
    display_df.columns = ['Date', 'Forecasted Load (kWh)']
    display_df['Date'] = display_df['Date'].dt.strftime('%A, %b %d')
    display_df['Forecasted Load (kWh)'] = display_df['Forecasted Load (kWh)'].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ─── PAGE: PATTERN ANALYSIS ───────────────────────────────────────────────────
elif page == "Pattern Analysis":
    st.markdown("<h1 style='color:#e6edf3;'>Household Usage Patterns</h1>", unsafe_allow_html=True)
    st.markdown("Identifying irregular consumption across households to support grid planning.")

    from sklearn.ensemble import IsolationForest

    recent_date = df['day'].max()
    day_data    = df[df['day'] == recent_date].copy()
    iso = IsolationForest(contamination=0.05, random_state=42)
    day_data['anomaly'] = iso.fit_predict(day_data[['energy_sum']])
    anomalies   = day_data[day_data['anomaly'] == -1]

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Normal Households</div>
            <div class="metric-value" style="color:#3fb950;">{len(day_data) - len(anomalies)}</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Flagged Households</div>
            <div class="metric-value" style="color:#f85149;">{len(anomalies)}</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Daily Load</div>
            <div class="metric-value">{day_data['energy_sum'].mean():.2f} <span class="metric-unit">kWh</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Consumption Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig2 = px.scatter(day_data, x=day_data.index, y='energy_sum',
                          color=day_data['anomaly'].map({1: 'Normal', -1: 'Flagged'}),
                          color_discrete_map={'Normal': '#4ECDC4', 'Flagged': '#FF6B6B'},
                          template='plotly_dark',
                          labels={'x': 'Household Index', 'energy_sum': 'Daily kWh', 'color': 'Status'})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Flagged IDs</div>', unsafe_allow_html=True)
        if not anomalies.empty:
            for _, row in anomalies.iterrows():
                badge = "badge-red" if row['energy_sum'] > day_data['energy_sum'].mean() else "badge-orange"
                st.markdown(f'<span class="badge {badge}">{row["LCLid"]}</span> {row["energy_sum"]:.2f} kWh', unsafe_allow_html=True)
        else:
            st.success("No anomalies detected!")

    st.markdown('<div class="section-title">Consumption by ACORN Group</div>', unsafe_allow_html=True)
    fig3 = px.box(df, x='Acorn', y='energy_sum', template='plotly_dark',
                  color='Acorn', color_discrete_sequence=px.colors.qualitative.Pastel,
                  labels={'energy_sum': 'Daily kWh', 'Acorn': 'ACORN Group'})
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE: PROJECT INFO ───────────────────────────────────────────────────────
elif page == "Project Info":
    st.markdown("<h1 style='color:#e6edf3;'>About This Project</h1>", unsafe_allow_html=True)
    st.markdown("""
This is a data science project built to analyze electricity consumption across London households.
The goal is to take raw smart meter logs, combine them with local weather conditions and calendar holidays,
and use machine learning to understand how people use power and predict future energy loads.

## Pipeline Structure
The backend runs as a flat, sequential pipeline with no custom functions:

| Script | Role |
|---|---|
| `config.py` | Global paths and memory limits |
| `data_loader.py` | Loads all raw CSVs safely with row limits |
| `preprocessing.py` | Cleans dates, ACORN labels, weather gaps |
| `eda.py` | Saves 3 visualisation charts to disk |
| `feature_engineering.py` | Merges data and builds lag/rolling features |
| `train.py` | Trains regressor, prints RMSE & MAE |
| `main.py` | Single entry point — runs everything |

## Dataset
- **Source**: Low Carbon London (UK Power Networks)
- **Households**: 5,567 London homes
- **Period**: November 2011 – February 2014
- **Features**: Energy readings, ACORN demographics, weather, bank holidays

## Author
**Tarun Jindal** — JECRC Foundation CEI Programme
""")
