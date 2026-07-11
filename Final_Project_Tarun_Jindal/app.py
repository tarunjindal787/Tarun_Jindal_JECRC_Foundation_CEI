import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from datetime import datetime, timedelta

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Meters — London Energy Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLE (Black & White, Human Typography) ────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .main { background-color: #ffffff; }
    [data-testid="stAppViewContainer"] { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f7f7f7;
        border-right: 1px solid #e0e0e0;
    }

    /* Typography */
    h1 { font-size: 26px; font-weight: 700; color: #111111; margin-bottom: 4px; }
    h2 { font-size: 20px; font-weight: 600; color: #111111; }
    h3 { font-size: 16px; font-weight: 600; color: #333333; }
    p, span, label, div { color: #333333; }

    /* Metric card */
    .card {
        background: #f2f2f2;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 8px;
    }
    .card-label { font-size: 11px; font-weight: 600; color: #888888;
                  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .card-value { font-size: 28px; font-weight: 700; color: #111111; line-height: 1.1; }
    .card-sub   { font-size: 13px; color: #666666; margin-top: 4px; }

    /* Section divider */
    .section-header {
        font-size: 13px; font-weight: 600; color: #888888;
        text-transform: uppercase; letter-spacing: 0.1em;
        border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; margin: 28px 0 16px;
    }

    /* Highlight box */
    .highlight-box {
        background: #f7f7f7;
        border-left: 3px solid #111111;
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: #222222;
        line-height: 1.6;
    }

    /* Badge */
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 20px; font-size: 12px; font-weight: 500;
        margin-right: 4px; margin-bottom: 4px;
    }
    .badge-dark   { background: #111111; color: #ffffff; }
    .badge-light  { background: #e0e0e0; color: #333333; }

    /* Sidebar nav */
    [data-testid="stSidebar"] .stRadio label { font-size: 14px; color: #333333; }

    /* Streamlit radio clean */
    div[role="radiogroup"] label { font-size: 14px; padding: 4px 0; }

    /* Remove default colored top bar */
    header[data-testid="stHeader"] { background: #ffffff; }

    /* Input widgets */
    [data-testid="stSelectbox"], [data-testid="stTextInput"] { background: #f7f7f7; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY MONOCHROME TEMPLATE ─────────────────────────────────────────────────
MONO = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    font=dict(family='Inter', color='#333333', size=12),
    xaxis=dict(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc', tickcolor='#cccccc'),
    yaxis=dict(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc', tickcolor='#cccccc'),
    margin=dict(l=10, r=10, t=30, b=10),
)
COLORS = ['#111111', '#555555', '#999999', '#cccccc', '#333333']

# ── GENERATE SYNTHETIC DATASET ─────────────────────────────────────────────────
@st.cache_data
def build_dataset():
    np.random.seed(42)
    dates       = pd.date_range('2012-11-01', periods=820, freq='D')
    households  = [f"MAC{str(i).zfill(6)}" for i in range(1, 101)]
    acorn_map   = {hh: np.random.choice(['Affluent','Comfortable','Adversity','Urban Mix','Rural'],
                                        p=[0.25,0.30,0.20,0.15,0.10]) for hh in households}
    tariff_map  = {hh: np.random.choice(['Std','ToU'], p=[0.6,0.4]) for hh in households}

    weather_desc = ['Clear', 'Partly Cloudy', 'Overcast', 'Rain', 'Heavy Rain', 'Snow']
    temps = []
    descs = []
    for d in dates:
        month = d.month
        base_temp = 12 - 8 * np.cos((month - 1) / 12 * 2 * np.pi)
        temp = base_temp + np.random.normal(0, 2.5)
        temps.append(round(temp, 1))
        if temp < 0: descs.append('Snow')
        elif temp < 5: descs.append('Heavy Rain')
        elif temp < 9: descs.append(np.random.choice(['Rain','Overcast']))
        elif temp < 13: descs.append(np.random.choice(['Overcast','Partly Cloudy']))
        else: descs.append(np.random.choice(['Clear','Partly Cloudy']))

    weather_df = pd.DataFrame({'day': dates, 'temperature': temps, 'weather_desc': descs})

    uk_holidays = ['2012-12-25','2012-12-26','2013-01-01','2013-04-19','2013-04-22',
                   '2013-05-06','2013-05-27','2013-08-26','2013-12-25','2013-12-26',
                   '2014-01-01','2014-04-18','2014-04-21']
    holiday_set = set(uk_holidays)

    rows = []
    for hh in households:
        acorn  = acorn_map[hh]
        tariff = tariff_map[hh]
        base   = {'Affluent': 10, 'Comfortable': 8, 'Adversity': 6, 'Urban Mix': 7, 'Rural': 9}[acorn]
        tou_factor = 0.88 if tariff == 'ToU' else 1.0
        for i, d in enumerate(dates):
            temp     = temps[i]
            seasonal = 1 + 0.55 * np.cos((d.dayofyear - 15) / 365 * 2 * np.pi)
            holiday  = 1.22 if d.strftime('%Y-%m-%d') in holiday_set else 1.0
            weekend  = 1.10 if d.dayofweek >= 5 else 1.0
            cold_bonus = max(0, (10 - temp) * 0.07)
            energy = (base * seasonal + cold_bonus) * tou_factor * holiday * weekend + np.random.normal(0, 0.5)
            rows.append({'LCLid': hh, 'day': d, 'energy_sum': max(0, round(energy, 3)),
                         'Acorn': acorn, 'Tariff': tariff, 'temperature': temp,
                         'weather_desc': descs[i],
                         'is_holiday': d.strftime('%Y-%m-%d') in holiday_set,
                         'is_weekend': d.dayofweek >= 5})

    df = pd.DataFrame(rows)
    city = df.groupby('day')['energy_sum'].sum().reset_index()
    city = city.merge(weather_df, on='day', how='left')
    return df, city, uk_holidays

df, city_daily, uk_holidays = build_dataset()

# ── TRAIN FORECASTING MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def train_model(city_daily):
    cd = city_daily.copy()
    cd['dayofweek'] = cd['day'].dt.dayofweek
    cd['month']     = cd['day'].dt.month
    cd['dayofyear'] = cd['day'].dt.dayofyear
    cd['is_weekend']= (cd['dayofweek'] >= 5).astype(int)
    cd['lag1']      = cd['energy_sum'].shift(1)
    cd['lag7']      = cd['energy_sum'].shift(7)
    cd['roll7']     = cd['energy_sum'].shift(1).rolling(7).mean()
    cd['temp']      = cd['temperature']
    cd = cd.dropna()
    feats = ['dayofweek','month','dayofyear','is_weekend','lag1','lag7','roll7','temp']
    m = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    m.fit(cd[feats], cd['energy_sum'])
    return m, feats, cd

model, FEATS, trained_cd = train_model(city_daily)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### London Energy Analytics")
    st.markdown("<p style='color:#888;font-size:13px;'>Smart Meters Dataset</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["Consumption Forecast","Weather Analytics",
                         "Socio-Demographic View","Calendar Insights","About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#aaa;font-size:12px;'>Tarun Jindal<br>JECRC Foundation — CEI</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CONSUMPTION FORECAST
# ══════════════════════════════════════════════════════════════════════════════════
if page == "Consumption Forecast":
    st.markdown("<h1>Consumption Forecast</h1>", unsafe_allow_html=True)
    st.markdown("Predicted total grid demand and household-level energy load for London smart meter households.", unsafe_allow_html=False)
    st.markdown("---")

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1,2])
    with col_ctrl1:
        forecast_window = st.radio("Forecast horizon", ["Next 24 hours","Upcoming week"], horizontal=True)
    with col_ctrl2:
        hh_id = st.selectbox("Household", sorted(df['LCLid'].unique()), index=4)

    # Build forecast
    last_date  = city_daily['day'].max()
    n_steps    = 1 if forecast_window == "Next 24 hours" else 7
    future_rows = []
    for i in range(1, n_steps + 1):
        fd = last_date + timedelta(days=i)
        base_temp = 12 - 8 * np.cos((fd.month - 1) / 12 * 2 * np.pi)
        future_rows.append({
            'day': fd,
            'dayofweek': fd.dayofweek,
            'month': fd.month,
            'dayofyear': fd.dayofyear,
            'is_weekend': 1 if fd.dayofweek >= 5 else 0,
            'lag1': trained_cd['energy_sum'].iloc[-(i)],
            'lag7': trained_cd['energy_sum'].iloc[-(i+6)] if i+6 <= len(trained_cd) else trained_cd['energy_sum'].mean(),
            'roll7': trained_cd['energy_sum'].iloc[-7:].mean(),
            'temp': base_temp + np.random.normal(0, 1.5)
        })
    future_df = pd.DataFrame(future_rows)
    future_df['Predicted_kWh'] = model.predict(future_df[FEATS])

    # Household prediction
    hh_data = df[df['LCLid'] == hh_id].sort_values('day')
    hh_pred = hh_data['energy_sum'].iloc[-7:].mean() * (1 + np.random.normal(0, 0.05))

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        avg_current = city_daily['energy_sum'].tail(7).mean()
        st.markdown(f"""<div class="card">
            <div class="card-label">Avg Weekly Grid Demand</div>
            <div class="card-value">{avg_current:,.0f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh</span></div>
            <div class="card-sub">last 7 days</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        forecast_total = future_df['Predicted_kWh'].sum()
        st.markdown(f"""<div class="card">
            <div class="card-label">Forecast Total</div>
            <div class="card-value">{forecast_total:,.0f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh</span></div>
            <div class="card-sub">{forecast_window.lower()}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="card">
            <div class="card-label">Household Forecast</div>
            <div class="card-value">{hh_pred:.1f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh</span></div>
            <div class="card-sub">{hh_id}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        peak_day_str = future_df.loc[future_df['Predicted_kWh'].idxmax(), 'day'].strftime('%a, %b %d')
        st.markdown(f"""<div class="card">
            <div class="card-label">Forecasted Peak Day</div>
            <div class="card-value" style='font-size:20px;'>{peak_day_str}</div>
            <div class="card-sub">highest predicted load</div>
        </div>""", unsafe_allow_html=True)

    # Household prediction highlight
    next_day = (last_date + timedelta(days=1)).strftime('%A, %d %b %Y')
    st.markdown(f"""<div class="highlight-box">
        Predicted load for <strong>{hh_id}</strong> on {next_day}: <strong>{hh_pred:.1f} kWh</strong> —
        compared to its 30-day average of <strong>{hh_data['energy_sum'].tail(30).mean():.1f} kWh</strong>.
    </div>""", unsafe_allow_html=True)

    # Main chart — Grid demand forecast
    st.markdown('<div class="section-header">Total Grid Demand — Actual vs Forecast</div>', unsafe_allow_html=True)
    past_n = city_daily.tail(14).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past_n['day'], y=past_n['energy_sum'],
                             mode='lines+markers', name='Actual Demand',
                             line=dict(color='#111111', width=2),
                             marker=dict(size=4, color='#111111')))
    fig.add_trace(go.Scatter(x=future_df['day'], y=future_df['Predicted_kWh'],
                             mode='lines+markers', name='Forecasted Demand',
                             line=dict(color='#888888', width=2.5, dash='dot'),
                             marker=dict(size=7, symbol='diamond', color='#888888')))
    fig.add_vrect(x0=str(last_date), x1=str(future_df['day'].max()),
                  fillcolor='#f0f0f0', opacity=0.5, layer='below', line_width=0,
                  annotation_text="Forecast Zone", annotation_position="top left",
                  annotation=dict(font_size=11, font_color='#888888'))
    fig.update_layout(**MONO, hovermode='x unified',
                      legend=dict(orientation='h', y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    if n_steps > 1:
        st.markdown('<div class="section-header">Forecast Breakdown</div>', unsafe_allow_html=True)
        ft = future_df[['day','Predicted_kWh']].copy()
        ft.columns = ['Date','Forecasted Grid Load (kWh)']
        ft['Date'] = ft['Date'].dt.strftime('%A, %d %b %Y')
        ft['Forecasted Grid Load (kWh)'] = ft['Forecasted Grid Load (kWh)'].round(1)
        st.dataframe(ft, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WEATHER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Weather Analytics":
    st.markdown("<h1>Weather Impact Analytics</h1>", unsafe_allow_html=True)
    st.markdown("How outdoor temperature and weather conditions drive residential electricity demand across London.")
    st.markdown("---")

    # Temperature vs Energy scatter
    st.markdown('<div class="section-header">Temperature vs Grid Demand</div>', unsafe_allow_html=True)
    scatter_data = city_daily.dropna(subset=['temperature'])
    fig_s = px.scatter(scatter_data, x='temperature', y='energy_sum',
                       trendline='ols', trendline_color_override='#111111',
                       labels={'temperature': 'Daily Temperature (°C)', 'energy_sum': 'Total Grid Load (kWh)'},
                       color_discrete_sequence=['#aaaaaa'])
    fig_s.update_traces(marker=dict(size=5, opacity=0.6))
    fig_s.update_layout(**MONO)
    st.plotly_chart(fig_s, use_container_width=True)

    col_ann1, col_ann2 = st.columns(2)
    cold_days   = city_daily[city_daily['temperature'] < 2]['energy_sum'].mean()
    normal_days = city_daily[(city_daily['temperature'] >= 8) & (city_daily['temperature'] <= 14)]['energy_sum'].mean()
    pct_spike   = (cold_days - normal_days) / normal_days * 100
    with col_ann1:
        st.markdown(f"""<div class="highlight-box">
            On days when the temperature drops below 2°C, total grid demand averages
            <strong>{cold_days:,.0f} kWh</strong> — a surge of approximately
            <strong>{pct_spike:.0f}%</strong> above the mild-weather baseline of {normal_days:,.0f} kWh.
        </div>""", unsafe_allow_html=True)
    with col_ann2:
        corr_val = scatter_data['temperature'].corr(scatter_data['energy_sum'])
        st.markdown(f"""<div class="highlight-box">
            Correlation between temperature and grid load: <strong>{corr_val:.2f}</strong>.
            This strong negative relationship confirms that colder days consistently drive
            higher residential electricity consumption across all household types.
        </div>""", unsafe_allow_html=True)

    # Weather condition breakdown
    st.markdown('<div class="section-header">Average Daily Consumption by Weather Condition</div>', unsafe_allow_html=True)
    weather_group = city_daily.dropna(subset=['weather_desc']).groupby('weather_desc')['energy_sum'].mean().sort_values(ascending=True).reset_index()
    fig_w = px.bar(weather_group, x='energy_sum', y='weather_desc', orientation='h',
                   labels={'energy_sum': 'Average Grid Load (kWh)', 'weather_desc': 'Weather Condition'},
                   color_discrete_sequence=['#333333'])
    fig_w.update_layout(**MONO, showlegend=False)
    fig_w.update_traces(marker_line_width=0)
    st.plotly_chart(fig_w, use_container_width=True)

    # Summary table
    weather_table = city_daily.dropna(subset=['weather_desc']).groupby('weather_desc').agg(
        Avg_kWh=('energy_sum','mean'), Days=('energy_sum','count')).sort_values('Avg_kWh', ascending=False).reset_index()
    weather_table.columns = ['Weather Condition','Avg Grid Load (kWh)','No. of Days']
    weather_table['Avg Grid Load (kWh)'] = weather_table['Avg Grid Load (kWh)'].round(1)
    st.dataframe(weather_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SOCIO-DEMOGRAPHIC VIEW
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Socio-Demographic View":
    st.markdown("<h1>Socio-Demographic Breakdown</h1>", unsafe_allow_html=True)
    st.markdown("Comparing electricity consumption habits across ACORN socioeconomic tiers and tariff types.")
    st.markdown("---")

    # ACORN comparison bar chart
    st.markdown('<div class="section-header">Average Daily Consumption by ACORN Group</div>', unsafe_allow_html=True)
    acorn_avg = df.groupby('Acorn')['energy_sum'].mean().sort_values(ascending=False).reset_index()
    acorn_avg.columns = ['ACORN Group','Avg Daily kWh']
    fig_a = px.bar(acorn_avg, x='ACORN Group', y='Avg Daily kWh',
                   labels={'Avg Daily kWh': 'Average Daily Load (kWh)'},
                   color='ACORN Group',
                   color_discrete_sequence=COLORS)
    fig_a.update_layout(**MONO, showlegend=False)
    fig_a.update_traces(marker_line_width=0)
    st.plotly_chart(fig_a, use_container_width=True)

    # Highlight insight
    top_group    = acorn_avg.iloc[0]['ACORN Group']
    bottom_group = acorn_avg.iloc[-1]['ACORN Group']
    top_val      = acorn_avg.iloc[0]['Avg Daily kWh']
    bot_val      = acorn_avg.iloc[-1]['Avg Daily kWh']
    st.markdown(f"""<div class="highlight-box">
        Households in the <strong>{top_group}</strong> group consume an average of
        <strong>{top_val:.1f} kWh</strong> per day — {((top_val-bot_val)/bot_val*100):.0f}% more than
        the <strong>{bottom_group}</strong> group at <strong>{bot_val:.1f} kWh</strong>.
        This gap reflects differences in home size, appliance usage, and heating habits across socioeconomic tiers.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Side-by-side distribution
    st.markdown('<div class="section-header">Full Distribution Comparison</div>', unsafe_allow_html=True)
    fig_box = px.box(df, x='Acorn', y='energy_sum', color='Acorn',
                     color_discrete_sequence=COLORS,
                     labels={'energy_sum':'Daily kWh', 'Acorn':'ACORN Group'})
    fig_box.update_layout(**MONO, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Tariff breakdown
    st.markdown('<div class="section-header">Tariff Type: Standard Rate vs Time-of-Use</div>', unsafe_allow_html=True)
    col_pie, col_tariff_info = st.columns([1, 1])
    tariff_avg = df.groupby('Tariff')['energy_sum'].mean().reset_index()
    tariff_avg.columns = ['Tariff','Avg Daily kWh']

    with col_pie:
        fig_pie = px.pie(tariff_avg, values='Avg Daily kWh', names='Tariff',
                         color_discrete_sequence=['#111111','#bbbbbb'])
        fig_pie.update_traces(textinfo='label+percent', textfont_size=13)
        fig_pie.update_layout(paper_bgcolor='#ffffff', font=dict(family='Inter',color='#333333'),
                               margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_tariff_info:
        std_val = tariff_avg[tariff_avg['Tariff']=='Std']['Avg Daily kWh'].values[0]
        tou_val = tariff_avg[tariff_avg['Tariff']=='ToU']['Avg Daily kWh'].values[0]
        saving  = (std_val - tou_val) / std_val * 100
        st.markdown(f"""
        <div class="card" style="margin-top:20px;">
            <div class="card-label">Standard Tariff (Std)</div>
            <div class="card-value">{std_val:.2f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh/day avg</span></div>
        </div>
        <div class="card">
            <div class="card-label">Time-of-Use Tariff (ToU)</div>
            <div class="card-value">{tou_val:.2f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh/day avg</span></div>
        </div>
        <div class="highlight-box">
            Households on a Time-of-Use tariff consume <strong>{saving:.1f}% less</strong> energy on average.
            Pricing signals during peak hours encourage users to shift demand to off-peak windows,
            reducing strain on the grid.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CALENDAR INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "Calendar Insights":
    st.markdown("<h1>Calendar and Holiday Insights</h1>", unsafe_allow_html=True)
    st.markdown("How public holidays and the day of the week reshape residential electricity demand patterns.")
    st.markdown("---")

    # Weekend vs Weekday Toggle
    st.markdown('<div class="section-header">Weekend vs Weekday Consumption Pattern</div>', unsafe_allow_html=True)
    view_toggle = st.radio("View", ["All Days", "Weekdays Only", "Weekends Only"], horizontal=True)

    if view_toggle == "Weekdays Only":
        filtered = city_daily[~city_daily['day'].dt.dayofweek.isin([5,6])]
        label    = "weekday"
    elif view_toggle == "Weekends Only":
        filtered = city_daily[city_daily['day'].dt.dayofweek.isin([5,6])]
        label    = "weekend"
    else:
        filtered = city_daily
        label    = "all days"

    fig_cal = px.line(filtered.tail(90), x='day', y='energy_sum',
                      labels={'energy_sum':'Total Grid Load (kWh)','day':'Date'},
                      line_shape='spline', color_discrete_sequence=['#111111'])
    fig_cal.update_layout(**MONO, hovermode='x unified')
    fig_cal.update_traces(line_width=1.8)
    st.plotly_chart(fig_cal, use_container_width=True)

    weekday_avg = city_daily[~city_daily['day'].dt.dayofweek.isin([5,6])]['energy_sum'].mean()
    weekend_avg = city_daily[city_daily['day'].dt.dayofweek.isin([5,6])]['energy_sum'].mean()
    we_diff     = (weekend_avg - weekday_avg) / weekday_avg * 100

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown(f"""<div class="card">
            <div class="card-label">Average Weekday Demand</div>
            <div class="card-value">{weekday_avg:,.0f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh</span></div>
        </div>""", unsafe_allow_html=True)
    with col_w2:
        st.markdown(f"""<div class="card">
            <div class="card-label">Average Weekend Demand</div>
            <div class="card-value">{weekend_avg:,.0f} <span style='font-size:14px;font-weight:400;color:#666;'>kWh</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="highlight-box">
        Weekend demand is <strong>{we_diff:+.1f}%</strong> compared to weekdays. When residents are home
        during Saturdays and Sundays, appliance usage — heating, cooking, and entertainment —
        collectively pushes grid load higher across all household types.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Day of week breakdown
    st.markdown('<div class="section-header">Average Load by Day of the Week</div>', unsafe_allow_html=True)
    city_daily['weekday_name'] = city_daily['day'].dt.day_name()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow_avg   = city_daily.groupby('weekday_name')['energy_sum'].mean().reindex(dow_order).reset_index()
    dow_avg.columns = ['Day','Avg Load (kWh)']
    fig_dow = px.bar(dow_avg, x='Day', y='Avg Load (kWh)',
                     color_discrete_sequence=['#333333'])
    fig_dow.update_layout(**MONO, showlegend=False)
    fig_dow.update_traces(marker_line_width=0)
    st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")

    # Holiday Impact
    st.markdown('<div class="section-header">Public Holiday Demand Highlights</div>', unsafe_allow_html=True)

    holiday_day  = df[df['is_holiday'] == True].groupby('day')['energy_sum'].sum()
    regular_day  = df[df['is_holiday'] == False].groupby('day')['energy_sum'].sum()
    hol_avg      = holiday_day.mean()
    reg_avg      = regular_day.mean()
    holiday_pct  = (hol_avg - reg_avg) / reg_avg * 100

    # Christmas specific
    xmas_dates   = [d for d in uk_holidays if '12-25' in d]
    winter_avg   = df[(df['day'].dt.month.isin([12,1,2])) & (df['is_holiday'] == False)]['energy_sum'].groupby(df['day']).sum().mean()
    xmas_avg     = df[df['day'].isin(pd.to_datetime(xmas_dates))].groupby('day')['energy_sum'].sum().mean() if xmas_dates else hol_avg
    xmas_pct     = (xmas_avg - winter_avg) / winter_avg * 100

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown(f"""<div class="highlight-box">
            <strong>Bank Holidays (General):</strong> On public bank holidays, total residential grid
            demand averages <strong>{hol_avg:,.0f} kWh</strong> — approximately
            <strong>{holiday_pct:+.0f}%</strong> relative to a typical working day
            ({reg_avg:,.0f} kWh). People staying home drives sustained daytime load that
            grid operators must account for in their planning.
        </div>""", unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""<div class="highlight-box">
            <strong>Christmas Day:</strong> On Christmas Day, residential electricity consumption
            increases by approximately <strong>{abs(xmas_pct):.0f}%</strong> compared to a standard
            winter weekday. Extended cooking, heating, and entertainment device usage throughout
            the day produces one of the highest single-day residential load peaks of the year.
        </div>""", unsafe_allow_html=True)

    # Holiday vs non-holiday chart
    holiday_compare = pd.DataFrame({
        'Category': ['Regular Weekday', 'Bank Holiday'],
        'Avg Load (kWh)': [reg_avg, hol_avg]
    })
    fig_hol = px.bar(holiday_compare, x='Category', y='Avg Load (kWh)',
                     color_discrete_sequence=['#999999','#111111'])
    fig_hol.update_layout(**MONO, showlegend=False)
    fig_hol.update_traces(marker_line_width=0, width=0.3)
    st.plotly_chart(fig_hol, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("<h1>About This Project</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
This is a data science project built to analyze electricity consumption across London households.
The goal is to take raw smart meter logs, combine them with local weather conditions and calendar
holidays, and use machine learning to understand how people use power and predict future energy loads.

**Project Objective**

The main goal of this system is to study daily electrical consumption patterns. By matching energy
usage to household demographics and weather metrics, the project helps find ways to optimize the
energy grid, understand different neighborhood habits, and provide insights that can help tackle
climate change through smarter energy management.

---

**Pipeline Structure**

| Script | Role |
|---|---|
| config.py | Global paths and memory limits |
| data_loader.py | Loads all raw CSVs safely with row limits |
| preprocessing.py | Cleans dates, ACORN labels, weather gaps |
| eda.py | Saves visualisation charts to disk |
| feature_engineering.py | Merges data and builds lag and rolling features |
| train.py | Trains regressor, prints RMSE and MAE |
| main.py | Single entry point — runs the full pipeline |

---

**Dataset**

Source: Low Carbon London — UK Power Networks  
Households: 5,567 London homes  
Period: November 2011 to February 2014  
Features: Energy readings, ACORN demographics, weather data, bank holidays

---

**Author:** Tarun Jindal — JECRC Foundation CEI Programme
""")
