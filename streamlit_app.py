# Streamlit app — thin orchestration layer
from pathlib import Path
from io import StringIO
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import DATA_PATH, ARTIFACT_PATH, TARGET, WINDOW
from core.data import DataLoader, make_splits
from core.models import AssetManager, HybridForecaster
from core.metrics import skill_horizon_expanding, plot_skill_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Hybrid Forecast (XGB + LSTM)", layout="wide")
st.title("📈 XGBoost + LSTM Hybrid — Demand Forecast")
st.caption("Loads engineered data, runs walk-forward forecast, shows RMSE/MAE/RAE and a planning horizon skill curve.")

# Sidebar
st.sidebar.header("Settings")
data_src = st.sidebar.radio("Data source", ["Use default path", "Upload CSV"], index=0)
default_data_path = Path(os.environ.get("DATA_PATH", str(DATA_PATH)))
artifact_path = Path(os.environ.get("ARTIFACT_PATH", str(ARTIFACT_PATH)))
test_ratio = st.sidebar.slider("Test ratio", 0.05, 0.30, 0.15, 0.01)
baseline = st.sidebar.selectbox("Skill baseline", ["persistence", "seasonal_7"], index=1)
threshold = st.sidebar.number_input("Skill threshold", value=0.0, step=0.05, format="%.2f")
run_btn = st.sidebar.button("Run forecast")

@st.cache_data(show_spinner=False)
def _load_raw_df_from_path(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def _engineer_from_df(df: pd.DataFrame) -> pd.DataFrame:
    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return DataLoader.load_csv(buf)

@st.cache_resource(show_spinner=False)
def _load_assets(path: Path) -> dict:
    return AssetManager(path).load()

assets = _load_assets(artifact_path)

def _plot_quartile_errors(df) -> go.Figure:
    """
    Plot RMSE and MAE for each advertising-spend quartile (Q1–Q4) using Plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'demand' (actuals)
        - 'y_hat'  (predictions)
        - one-hot flags 'tot_adv_exp_i_1_Q1' … 'Q4'
    """
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['quartile'], y=df['RMSE'].values, name='RMSE',
        marker_color='steelblue', text=[f'{v:.1f}' for v in df['RMSE'].values],
        textposition='outside'))

    fig.add_trace(go.Bar(
        x=df.quartile, y=df['MAE'].values, name='MAE',
        marker_color='orange', text=[f'{v:.1f}' for v in df['MAE'].values],
        textposition='outside'))

    fig.update_layout(
        title='XGB + LSTM Hybrid Quartile Distribution (FH=1) : RMSE and MAE',
        yaxis_title='Error (units)',
        barmode='group', template='plotly_white', height=500)
    return fig


def _plot_timeseries(df_plot: pd.DataFrame, target_col: str = TARGET) -> go.Figure:
    fig = go.Figure()
    #Actual values
    fig.add_trace(go.Scatter(
        x= df_plot.index, 
        y= df_plot[target_col], 
        mode='lines+markers', 
        name="Actual Values", 
        line=dict(color='green', width=3, dash='dot'),
        marker=dict(symbol='circle', size=5, opacity=0.7)
    ))

    # Predicted values FH=1 (increase visibility)
    fig.add_trace(go.Scatter(
        x= df_plot.index, 
        y= df_plot['y_hat'], 
        mode='lines', 
        name="Hybrid Model FH=1", 
        line=dict(color='red', dash='dash',width=2),  # Make line bolder
        #marker=dict(symbol='square', size=4)  # Increase marker size
    ))
    # Predicted values FH= Testing Set (Single Shot)
    fig.add_trace(go.Scatter(
        x= df_plot.index, 
        y= df_plot['y_fwd'], 
        mode='lines', 
        name="Hybrid Model FH = One Single Shot", 
        line=dict(color='purple', dash='dash', width=2),  # Make line bolder
        #marker=dict(symbol='square', size=3),  # Increase marker size
        opacity=0.45
    ))

    fig.update_layout(
        title="Actual vs Predicted Values Over Time",
        xaxis_title="Date",
        yaxis_title="Values",
        xaxis=dict(tickangle=45),
        legend=dict(orientation="h", yanchor="bottom",
            y=1.02, xanchor="left", x=0),
        hovermode="x unified",        
        template="plotly_white"
    )    
    # fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[target_col], name="Actual", mode="lines"))
    # fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["y_hat"], name="1 day Forecast ", mode="lines", line=dict(dash="dash")))
    # fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["y_fwd"], name="One Shot Forecast ", mode="lines", line=dict(dash="dash")))
    # fig.update_layout(title="Actual vs Forecast", xaxis_title="Date", yaxis_title="Units", template="plotly_white", hovermode="x unified")
    return fig

def _plot_rae(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['rae'],
        mode='lines+markers',
        name='RAE (%)',
        line=dict(color='orange'),
        marker=dict(size=4)
    ))
   # Add reference line (optional, e.g. 10% RAE threshold)
    fig.add_hline(
        y=0.10,  # 10% RAE threshold
        line=dict(color='red', dash='dash'),
        annotation_text='10% RAE Threshold',
        annotation_position='top right'
    )
    rae_mean = df['rae'].mean()
    # Add reference line (optional, e.g. 10% RAE threshold)
    fig.add_hline(
        y= rae_mean,  # rae mean
        line=dict(color='green', dash='dot'),
        annotation_text=f'{100*rae_mean:.2f} RAE Mean FH =1d',
        annotation_position='bottom right'
    )

    fig.update_layout(
        title='Pointwise Relative Absolute Error (RAE) over 1 day Forecast Horizon',
        xaxis_title='Date',
        yaxis_title='RAE (%)',
        template='plotly_white',
        height=400
    )
    return fig

# Data choice
if data_src == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or switch to default path.")
        raw_df = None
else:
    if not default_data_path.exists():
        st.error(f"Default data not found at {default_data_path}. Upload a CSV instead.")
        raw_df = None
    else:
        raw_df = _load_raw_df_from_path(default_data_path)

if raw_df is not None:
    st.subheader("Data preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    if run_btn:
        with st.spinner("Engineering features & loading artefacts..."):
            try:
                df_full = _engineer_from_df(raw_df)
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")
                st.stop()

            try:
                assets = _load_assets(artifact_path)
            except Exception as e:
                st.error(f"Loading artefacts failed: {e}")
                st.info("Ensure ./classes contains: xgb.pkl, lstm.keras, qc.pkl, sx.pkl, sy.pkl, sr.pkl, features.pkl, bootstrap.csv")
                st.stop()

        # Split: dev (train+val) and test
        dev_ratio = 1 - 2 * test_ratio
        df_train, df_val, df_test = make_splits(df_full, ratios=(dev_ratio, test_ratio, test_ratio))
        df_dev = pd.concat([df_train, df_val])

        # Forecast
        try:
            forecaster = HybridForecaster(assets)
        except Exception as e:
            st.error(f"Could not initialise forecaster: {e}")
            st.stop()

        with st.spinner("Running walk-forward forecast..."):
            try:
                df_pred = forecaster.single_shot(df_test)
            except Exception as e:
                st.error(f"Forecast Horizon 1d failed: {e}")
                st.stop()
            try:
                df_pred_alt = forecaster.walk_forward(df_test)
            except Exception as e:
                st.error(f"One Shot prediction failed: {e}")
                st.stop()

        #join the dataframes
        df_test_results = pd.concat([df_pred[[TARGET, 'y_hat']],df_pred_alt[['y_fwd','quartile']]], axis=1, join="inner")  # display first row of both predictions side by side
        df_test_results['quartile'] = df_test_results['quartile'].str.upper() 
        # RAE metrics
        df_test_results['rae_fwd'] = df_test_results.apply(lambda row: abs(row['demand'] - row['y_fwd']) / row['demand'], axis=1)
        df_test_results['rae']     = df_test_results.apply(lambda row: abs(row['demand'] - row['y_hat']) / row['demand'], axis=1)        
        
        st.subheader("Results by Scenario")
        st.caption("1 day prediction Scenario")
        dict_result = df_pred.attrs
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{dict_result['rmse']:,.2f}")
        c2.metric("MAE", f"{dict_result['mae']:,.2f}")
        c3.metric("Mean RAE", f"{100*dict_result['mrae']:,.2f}%")        

        # Optional quartile summary
        if "quartile" in df_test_results.columns:
            st.subheader("Model performance by Quartile Scenario")
            rmse_f = lambda x: np.sqrt(mean_squared_error(x[TARGET], x["y_fwd"]))
            mae_f = lambda x: mean_absolute_error(x[TARGET], x["y_fwd"])
            df_quart = (
                df_test_results.groupby("quartile")
                .apply(lambda x: pd.Series({"RMSE": rmse_f(x), "MAE": mae_f(x)}))
                .reset_index()
                .sort_values("quartile")
            )

            st.plotly_chart(_plot_quartile_errors(df_quart))

        st.caption("One Shot prediction Scenario")
        dict_result = df_pred_alt.attrs
        c4, c5, c6 = st.columns(3)
        c4.metric("RMSE", f"{dict_result['rmse']:,.2f}")
        c5.metric("MAE", f"{dict_result['mae']:,.2f}")
        c6.metric("Mean RAE", f"{100*dict_result['mrae']:,.2f}%")


        st.subheader("Key Plots")
        # Plots
        st.plotly_chart(_plot_timeseries(df_test_results, TARGET), use_container_width=True)
        st.plotly_chart(_plot_rae(df_test_results), use_container_width=True)

        # Skill curve (seed with dev tail when seasonal baseline)
        L = 1 if baseline == "persistence" else int(baseline.split("_")[1])
        #history = df_dev[TARGET].iloc[-L:]
        try:
            h_days, skill_df = skill_horizon_expanding(
                y_true=df_test_results[TARGET], y_pred=df_test_results["y_fwd"],
                baseline=baseline, threshold=threshold, history=None
            )
            st.subheader(f"Planning horizon (H) Graph")#: {h_days} days")
            st.plotly_chart(plot_skill_curve(skill_df, h_days,
                                             title=f"Skill vs {baseline} baseline (expanding window)"),
                            use_container_width=True)
        except Exception as e:
            st.warning(f"Skill curve could not be computed: {e}")

        st.download_button(
            "Download forecast CSV",
            df_test_results[[TARGET, "y_fwd"]].to_csv().encode("utf-8"),
            file_name="forecast_walkforward.csv",
            mime="text/csv"
        )
