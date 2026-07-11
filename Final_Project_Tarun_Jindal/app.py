import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from data_pipeline import load_data, get_city_wide_daily
from train_models import train_forecasting_model, train_anomaly_model

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Energy Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1wivap2 {
        color: #FAFAFA !important;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_data
def load_all_data():
    df = load_data()
    city_daily = get_city_wide_daily(df)
    return df, city_daily

@st.cache_resource
def load_models():
    # If models don't exist, train them now
    if not os.path.exists(os.path.join(MODEL_DIR, "forecast_model.pkl")) or \
       not os.path.exists(os.path.join(MODEL_DIR, "anomaly_model.pkl")):
        with st.spinner("Training models for the first time. This may take a minute..."):
            train_forecasting_model()
            train_anomaly_model()
            
    try:
        with open(os.path.join(MODEL_DIR, "forecast_model.pkl"), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "anomaly_model.pkl"), 'rb') as f:
            iso_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "latest_date.txt"), 'r') as f:
            latest_date_str = f.read().strip()
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
        return rf_model, iso_model, latest_date
    except Exception as e:
        return None, None, None

def main():
    st.sidebar.title("⚡ Energy Analytics")
    st.sidebar.markdown("Smart Meter Data Analysis")
    
    page = st.sidebar.radio("Navigation", ["Overview", "Load Forecasting", "Pattern Analysis"])
    
    st.title("London Smart Meters Dashboard")
    
    with st.spinner("Loading data & models..."):
        df, city_daily = load_all_data()
        rf_model, iso_model, latest_date = load_models()
        
    if rf_model is None:
        st.warning("⚠️ Models not found. Please run `train_models.py` locally before deploying.")
        st.stop()

    if page == "Overview":
        st.header("Historical City-Wide Energy Consumption")
        st.markdown("Visualizing the aggregated daily energy consumption across all sampled households in London.")
        
        recent = city_daily.tail(60).copy()
        
        # Metric Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; color:#888;">Avg Daily Usage</p>
                <h2 style="margin:0;">{recent['energy(kWh/hh)'].mean():.2f} kWh</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; color:#888;">Peak Usage Day</p>
                <h2 style="margin:0;">{recent.loc[recent['energy(kWh/hh)'].idxmax(), 'date'].strftime('%b %d')}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; color:#888;">Households Tracked</p>
                <h2 style="margin:0;">{df['LCLid'].nunique()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")
        
        fig = px.line(recent, x='date', y='energy(kWh/hh)', 
                      title="Last 60 Days: Total Energy Consumption (kWh)",
                      template="plotly_dark",
                      line_shape='spline')
        fig.update_traces(line_color="#4ECDC4", line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Load Forecasting":
        st.header("7-Day Load Forecast")
        st.markdown("Statistical predictions for the next week of energy consumption based on historical patterns.")
        
        future_dates = [latest_date + timedelta(days=i) for i in range(1, 8)]
        future_df = pd.DataFrame({'date': future_dates})
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['dayofyear'] = future_df['date'].dt.dayofyear
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        
        predictions = rf_model.predict(future_df[['dayofweek', 'month', 'dayofyear', 'is_weekend']])
        future_df['Predicted_kWh'] = predictions
        
        fig = go.Figure()
        
        # Plot past data for context
        past_dates = city_daily.tail(14)
        fig.add_trace(go.Scatter(x=past_dates['date'], y=past_dates['energy(kWh/hh)'],
                                 mode='lines+markers', name='Actual',
                                 line=dict(color='#FAFAFA', width=2)))
                                 
        # Plot forecast
        fig.add_trace(go.Scatter(x=future_df['date'], y=future_df['Predicted_kWh'],
                                 mode='lines+markers', name='Forecast',
                                 line=dict(color='#FF6B6B', width=3, dash='dot')))
                                 
        fig.update_layout(title="Energy Forecast (Next 7 Days)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Pattern Analysis":
        st.header("Household Usage Patterns & Anomalies")
        st.markdown("Automatically flagging households with irregular energy consumption patterns to identify potential issues.")
        
        recent_date = df['tstp'].dt.date.max()
        recent_data = df[df['tstp'].dt.date == recent_date].copy()
        daily_hh = recent_data.groupby(['LCLid'])['energy(kWh/hh)'].sum().reset_index()
        
        daily_hh['anomaly'] = iso_model.predict(daily_hh[['energy(kWh/hh)']])
        anomalies = daily_hh[daily_hh['anomaly'] == -1]
        
        st.info(f"**Insights for {recent_date}**: Detected **{len(anomalies)}** anomalous households out of {len(daily_hh)} total tracked.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(daily_hh, x=daily_hh.index, y='energy(kWh/hh)', 
                             color=daily_hh['anomaly'].astype(str),
                             color_discrete_map={'-1': '#FF6B6B', '1': '#4ECDC4'},
                             title="Household Consumption Distribution",
                             labels={'x': "Household Index", 'energy(kWh/hh)': "Daily kWh"},
                             template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Anomalous Households")
            if not anomalies.empty:
                st.dataframe(anomalies[['LCLid', 'energy(kWh/hh)']].sort_values(by='energy(kWh/hh)', ascending=False), hide_index=True)
            else:
                st.success("No anomalies detected today!")

if __name__ == "__main__":
    main()
