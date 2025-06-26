import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
#import pandas_ta as ta

# --- App Configuration ---
st.set_page_config(
    page_title="QuantPattern Pro - Elite Edition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Core Logic Functions ---

def get_stock_data(ticker):
    """Fetches fresh stock data each time."""
    return yf.download(ticker, start="2000-01-01", end=datetime.now(), progress=False)

def add_technical_indicators(df):
    """Adds technical indicators without caching."""
    df['price_change'] = df['Adj Close'].pct_change()
    df.ta.rsi(length=14, append=True, col_names=('RSI',))
    df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD','MACDh','MACDs'))
    df.dropna(inplace=True)
    return df

def _z_normalize(series):
    return (series - np.mean(series)) / np.std(series)

def find_similar_patterns(data_to_search, target_pattern, method, tol):
    pattern_len = len(target_pattern)
    matched_indices = []
    matched_distances = []

    if method == "Z-Norm Euclidean":
        target_norm = _z_normalize(target_pattern)

    for i in range(len(data_to_search) - pattern_len + 1):
        window = data_to_search.iloc[i : i + pattern_len]
        distance = np.inf

        if method == "Cosine Similarity":
            distance = euclidean(target_pattern.flatten(), window.to_numpy().flatten())
        
        elif method == "Z-Norm Euclidean":
            window_norm = _z_normalize(window.to_numpy())
            distance = euclidean(target_norm.flatten(), window_norm.flatten())
        
        elif method == "Dynamic Time Warping (DTW)":
            total_dist = 0
            for col in range(target_pattern.shape[1]):
                dist, _ = fastdtw(target_pattern[:, col], window.iloc[:, col].values, dist=euclidean)
                total_dist += dist
            distance = total_dist / target_pattern.shape[1]

        if distance < tol:
            matched_indices.append(i + pattern_len - 1)
            matched_distances.append(distance)

    return matched_indices, matched_distances

def analyze_outcomes(data, indices, distances, period, cost_pct):
    outcomes = []
    for i, idx in enumerate(indices):
        if idx + period >= len(data): continue
        start_date = data.index[idx]
        start_price = data['Adj Close'].iloc[idx]
        outcome_slice = data['Adj Close'].iloc[idx : idx + period + 1]
        
        peak_profit_pct = (((outcome_slice.max() - start_price) / start_price) * 100) - cost_pct
        days_to_peak = (outcome_slice.idxmax() - start_date).days if peak_profit_pct > 0 else 0
        final_profit_pct = (((outcome_slice.iloc[-1] - start_price) / start_price) * 100) - cost_pct
        running_max = outcome_slice.cummax()
        max_drawdown = (((outcome_slice - running_max) / running_max).min()) * 100
        daily_profit_pct = (((outcome_slice / start_price) - 1) * 100).tolist()

        outcomes.append({
            'match_date': start_date, 'distance': distances[i], 'final_profit_pct': final_profit_pct,
            'peak_profit_pct': peak_profit_pct, 'days_to_peak': days_to_peak,
            'max_drawdown_pct': max_drawdown, 'daily_outcomes': daily_profit_pct
        })
    return pd.DataFrame(outcomes).sort_values(by='distance').reset_index(drop=True)

def plot_outcomes_with_confidence(df):
    fig = go.Figure()
    all_outcomes = np.array(df['daily_outcomes'].tolist())
    mean_outcome = np.mean(all_outcomes, axis=0)
    p25 = np.percentile(all_outcomes, 25, axis=0)
    p75 = np.percentile(all_outcomes, 75, axis=0)
    x = list(range(len(mean_outcome)))

    fig.add_trace(go.Scatter(x=x, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p25, mode='lines', line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(0,100,80,0.2)', name='25-75th Percentile'))
    fig.add_trace(go.Scatter(x=x, y=mean_outcome, mode='lines', line=dict(color='rgba(0,100,80,1)'), name='Avg. Outcome'))
    fig.add_hline(y=0, line_dash="dot", line_color="black")
    fig.update_layout(title="<b>Strategy Performance Distribution</b>",
                      xaxis_title="Days Since Match", yaxis_title="Profit / Loss (%)", hovermode="x unified")
    return fig

def plot_price_with_matches(df, target_start_date, target_end_date, match_dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Price', line=dict(color='lightgrey')))
    target_df = df.loc[target_start_date:target_end_date]
    fig.add_trace(go.Scatter(x=target_df.index, y=target_df['Adj Close'], mode='lines', line=dict(color='red', width=4), name='Target Pattern'))
    match_prices = df.loc[match_dates, 'Adj Close']
    fig.add_trace(go.Scatter(x=match_prices.index, y=match_prices.values, mode='markers',
                             marker=dict(color='blue', size=8, symbol='circle', opacity=0.7), name='Historical Matches'))
    fig.update_layout(title="<b>Price History with Pattern Matches</b>", xaxis_title="Date", yaxis_title="Adjusted Close Price")
    return fig

def run_analysis(params):
    full_data = get_stock_data(params['ticker'])
    if full_data is None or full_data.empty: return {"error": "Failed to fetch stock data."}

    feature_map = {'Price Change': 'price_change', 'RSI': 'RSI', 'MACD': 'MACDs'}
    selected_cols = [feature_map[feat] for feat in params['features']]
    data = add_technical_indicators(full_data.copy())

    try:
        end_loc = data.index.get_loc(params['date'], method='ffill')
        start_loc = end_loc - params['pattern_len'] + 1
        target_df = data.iloc[start_loc : end_loc + 1][selected_cols]
    except KeyError:
        return {"error": f"Selected date {params['date'].date()} not found in trading data."}
    if len(target_df) < params['pattern_len']: return {"error": "Not enough data to form pattern."}

    target_start_date, target_end_date = target_df.index[0], target_df.index[-1]
    historical_data = data[data.index < target_end_date][selected_cols]

    if params['match_method'] == "Cosine Similarity":
        scaler = StandardScaler()
        historical_data_to_search = pd.DataFrame(scaler.fit_transform(historical_data), index=historical_data.index, columns=selected_cols)
        target_pattern_to_search = scaler.transform(target_df)
    else:
        historical_data_to_search = historical_data
        target_pattern_to_search = target_df.to_numpy()

    relative_indices, distances = find_similar_patterns(
        historical_data_to_search, target_pattern_to_search, params['match_method'], params['tolerance'])

    if len(relative_indices) == 0:
        return {"error": "No similar historical patterns found. Try increasing tolerance or changing the algorithm."}
    
    original_indices = [data.index.get_loc(historical_data.index[i]) for i in relative_indices]
    outcomes_df = analyze_outcomes(data, original_indices, distances, params['analysis_period'], params['cost_pct'])

    return {
        "error": None, "outcomes_df": outcomes_df, "full_data": data, 
        "target_start_date": target_start_date, "target_end_date": target_end_date
    }

# --- Streamlit UI ---
st.title("QuantPattern Pro - Elite Edition")
st.caption("An advanced backtesting tool with DTW, Z-Normalization, and Drawdown Analysis.")

with st.sidebar:
    st.header("1. Analysis Setup")
    params = {}
    params['ticker'] = st.text_input("Stock Ticker", "MSFT").upper()
    params['date'] = st.date_input("Pattern End Date", pd.to_datetime("2024-05-01"))

    # Add Refresh Button
    if st.button("🔄 Refresh Stock Data"):
        st.cache_data.clear()
        st.experimental_rerun()

    st.header("2. Pattern Definition")
    params['features'] = st.multiselect("Features", ['Price Change', 'RSI', 'MACD'], ['Price Change', 'RSI'])
    params['pattern_len'] = st.slider("Pattern Length (Days)", 5, 40, 15)

    st.header("3. Matching Algorithm")
    params['match_method'] = st.selectbox(
        "Algorithm", 
        ['Dynamic Time Warping (DTW)', 'Z-Norm Euclidean', 'Cosine Similarity'],
        index=0,
        help="DTW: Good for similar shapes, Z-Norm: Good for normalized matching, Cosine: Good for direction"
    )

    if params['match_method'] == 'Dynamic Time Warping (DTW)':
        params['tolerance'] = st.slider("Tolerance", 0.1, 20.0, 5.0)
    elif params['match_method'] == 'Z-Norm Euclidean':
        params['tolerance'] = st.slider("Tolerance", 0.1, 5.0, 1.5)
    else:
        params['tolerance'] = st.slider("Tolerance", 0.01, 1.0, 0.2)

    st.header("4. Backtest Parameters")
    params['analysis_period'] = st.slider("Forward Analysis (Days)", 10, 120, 60)
    params['cost_pct'] = st.slider("Transaction Cost (%)", 0.0, 2.0, 0.1, step=0.05)

    run_button_text = "🚀 Run Analysis"
    if params['match_method'] == 'Dynamic Time Warping (DTW)':
        run_button_text += " (Slower)"
    run_button = st.button(run_button_text, use_container_width=True, type="primary")

if run_button:
    with st.spinner("Running Analysis..."):
        results = run_analysis(params)

    if results.get("error"):
        st.error(f"**Analysis Failed:** {results['error']}")
    else:
        outcomes_df = results['outcomes_df']
        st.success(f"Analysis Complete! Found **{len(outcomes_df)}** historical matches.")

        st.subheader("Strategy Performance & Risk Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg. Final Profit", f"{outcomes_df['final_profit_pct'].mean():.2f}%")
        col2.metric("Median Final Profit", f"{outcomes_df['final_profit_pct'].median():.2f}%")
        col3.metric("Win Rate (Peak > 0%)", f"{(outcomes_df['peak_profit_pct'] > 0).mean() * 100:.1f}%")
        col4.metric("Avg. Max Drawdown", f"{outcomes_df['max_drawdown_pct'].mean():.2f}%", delta_color="inverse")

        tab1, tab2, tab3 = st.tabs(["📊 Performance Chart", "📈 Price Chart Context", "📋 Top 5 Best Matches"])
        with tab1:
            st.plotly_chart(plot_outcomes_with_confidence(outcomes_df), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_price_with_matches(results['full_data'], results['target_start_date'], results['target_end_date'], outcomes_df['match_date']), use_container_width=True)
        with tab3:
            top_5 = outcomes_df.head(5)
            for _, row in top_5.iterrows():
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Match Date", f"{row['match_date'].strftime('%Y-%m-%d')}")
                    c2.metric("Similarity Distance", f"{row['distance']:.4f}")
                    c3.metric("Peak Profit", f"{row['peak_profit_pct']:.2f}%")
                    c4.metric("Max Drawdown", f"{row['max_drawdown_pct']:.2f}%")
else:
    st.info("Set your parameters in the sidebar, then click 'Run Analysis' to begin.")
